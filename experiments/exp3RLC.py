import pandas as pd
import glob
import os
from itertools import product
from argparse import Namespace
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import lightning as L
from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

# <Settings_>
fast_dev_run = False  # True: activate lightning dev run; False: Normal operation
max_epochs = 5
torch.set_float32_matmul_precision('high')
batchsize = 64
small=False
hole=False
train = False
test = True
# <_Settings>

# <functions>
def find_global_min_max(folder_path):
    # Initialize dictionaries to store the global min and max values for each column
    global_min = {}
    global_max = {}
    
    # List all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Iterate through each column in the DataFrame
        for column in df.columns:
            if column not in global_min:
                global_min[column] = df[column].min()
                global_max[column] = df[column].max()
            else:
                global_min[column] = min(global_min[column], df[column].min())
                global_max[column] = max(global_max[column], df[column].max())
    
    return global_min, global_max

def compute_absolute_max(global_min, global_max):
    absolute_max = {}
    for column in global_min:
        absolute_max[column] = max(abs(global_min[column]), abs(global_max[column]))
    return absolute_max

def concatenate_csv_files(folder_path, output_file, small=False, hole=False):
    # List to store dataframes
    dataframes = []
    
    # List all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if hole:
        # Skip the 6th, 7th, and 8th files (index 5, 6, 7 since indexing starts from 0)
        csv_files = [file for i, file in enumerate(csv_files) if i not in [5, 6, 7]]
    
    # Read and append each CSV file
    for file in csv_files:
        df = pd.read_csv(file)
        # If small is True, take only the first 10% of the data
        if small:
            df = df.iloc[:int(len(df) * 0.1)]
        dataframes.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined CSV saved to {output_file}")

def load_data(file_path: str, window_size: int, step_size: int, voltage_scaler, current_scaler) -> torch.tensor:
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(file_path)
    df['voltage'] = df['voltage'] / voltage_scaler
    df['current'] = df['current'] / current_scaler
    print("Original data shape:", df.shape)

    # Convert the DataFrame to a PyTorch tensor
    data_tensor = torch.tensor(df.values, dtype=torch.float32)

    # Create slices using a sliding window
    num_windows = (data_tensor.size(0) - window_size) // step_size + 1
    windows = [data_tensor[i*step_size:i*step_size+window_size] for i in range(num_windows)]
    sliced_data = torch.stack(windows)

    return sliced_data
# <functions>
# <classes>
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

class LitAutoEncoder(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.encoder = nn.Sequential(nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 25))
        self.decoder = nn.Sequential(nn.Linear(25, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 200))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch
        x = x.view(-1, 200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        #x_hat = x_hat.view(batchsize, 100, 2)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch
        x = x.view(-1, 200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log('epoch', self.current_epoch, prog_bar=False)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)

    def threshold_step(self, batch, batch_idx):
        x = batch
        x = x.view(-1, 200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log('threshold_loss', val_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch
        x = x.view(-1, 200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        log_values = {"test_loss": test_loss, "test_case": test_case}
        self.log(log_values, on_step=True, on_epoch=False)

    def forward(self, sample):
        x = sample.view(200)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss, sample, x_hat.view(100, 2)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
# <classes>

# <Callbacks>
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
    
class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Access the logged train loss
        train_loss = trainer.callback_metrics.get('train_loss')
        if train_loss is not None:
            self.train_loss.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Access the logged validation loss
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_loss.append(val_loss.item())    

# <Callbacks>

if __name__ == "__main__":
    if not os.path.exists('simulation/3RLC_Sim/data/train/3RLC_all.csv'):
        print('Creating file with merged data')
        concatenate_csv_files('simulation/3RLC_Sim/data/train', 'simulation/3RLC_Sim/data/train/3RLC_all.csv')
    
    if not os.path.exists('simulation/3RLC_Sim/data/train/3RLC_all_small.csv'):
        print('Creating file with merged data')
        concatenate_csv_files('simulation/3RLC_Sim/data/train', 'simulation/3RLC_Sim/data/train/3RLC_all_small.csv', small=True)
    
    if not os.path.exists('simulation/3RLC_Sim/data/train/3RLC_all_holes.csv'):
        print('Creating file with merged data')
        concatenate_csv_files('simulation/3RLC_Sim/data/train', 'simulation/3RLC_Sim/data/train/3RLC_all_holes.csv', hole=True)

    # Get Min Max Values for global scaling
    folder_path = 'simulation/3RLC_Sim/data/train'
    min_values, max_values = find_global_min_max(folder_path)
    print("Global Min Values:", min_values)
    print("Global Max Values:", max_values)

    absolute_max_values = compute_absolute_max(min_values, max_values)
    print("Absolute Max Values:", absolute_max_values)

    current_scaler = round(absolute_max_values['current'])
    voltage_scaler = round(absolute_max_values['voltage'])

    # Define hyperparameters and their ranges
    var_models = ['all', 'all_small', 'all_holes', '0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20']
    hidden_size = [20]
    torch_seed = [0]
    window_size = 100
    step_size = 100

    # Create all combinations of hyperparameters
    hparam_combinations = product(var_models, torch_seed)

    for var_model, torch_seed in hparam_combinations:
        # this loop goes through the experiments
        csv_logger = CSVLogger('3RLC_logs', name=f'my_model_{var_model}_{torch_seed}')
        #exp_results = [var_model, None, None, None, None, None, None, None, None, None]
        train_file = f'simulation/3RLC_Sim/data/train/3RLC_{var_model}.csv'
        train_set = load_data(train_file, window_size, step_size, voltage_scaler, current_scaler)
        # Scale Data
        print(train_set.shape)
        
        L.seed_everything(torch_seed, workers=True)
        seed = torch.Generator().manual_seed(torch_seed)

        train_set_size = int(train_set.shape[0] * 0.8)
        valid_set_size = train_set.shape[0] - train_set_size
        train_set, valid_set = torch.split(train_set, [train_set_size, valid_set_size], dim=0)
        #train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

        train_set = TimeSeriesDataset(train_set)
        valid_set = TimeSeriesDataset(valid_set)

        train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_set, num_workers=8)

        # Create a Namespace object for hyperparameters
        hparams = Namespace(
            hidden_size=hidden_size,
        )

        basic_checkpoint_callback = ModelCheckpoint(dirpath='./basic_checkpoints/')

        custom_checkpoint_callback = ModelCheckpoint(
            #every_n_epochs=1,
            save_top_k=-1,  # Keep all checkpoints
            monitor='val_loss',
            dirpath='./custom_checkpoints/',
            filename= f'model_{var_model}' + '{epoch:02d}_{val_loss:.4f}'
            )

        trainer = L.Trainer(callbacks=[custom_checkpoint_callback],
                            logger=csv_logger, limit_train_batches=100,
                            max_epochs=max_epochs, deterministic=True, fast_dev_run=fast_dev_run)

        # init model and trainer
        autoencoder = LitAutoEncoder(hparams)
        trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        custom_checkpoint_callback.best_model_path

        if test and not fast_dev_run:
            # Testing
            if var_model in ['all', 'all_small', 'all_holes']:
                test_cases = ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20']
            else:
                #test_cases = [var_model]
                test_cases = ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20']

            if not os.path.exists("./3RLC_filelogs/"):
                # Create the folder if it doesn't exist
                os.makedirs("./3RLC_filelogs/")
                print(f"Folder ./3RLC_filelogs/ created.")
            else:
                print(f"Folder ./3RLC_filelogs/ already exists.")

            # Model testing
            autoencoder.eval()
            with torch.no_grad():
                for test_case in test_cases:
                    logdir = f"./3RLC_filelogs/{var_model}_{test_case}"

                    # Test case normal system
                    test_ok_file = f'simulation/3RLC_Sim/data/test/3RLC_Test_{test_case}.csv'
                    test_ok_set = load_data(test_ok_file, window_size, step_size, voltage_scaler, current_scaler)
                    test_ok_set = TimeSeriesDataset(test_ok_set)
                    test_ok_loader = DataLoader(test_ok_set, batch_size=1, shuffle=False, num_workers=8)
                    ok_test_log = []
                    for idx in range(len(test_ok_set)):
                        sample = test_ok_set[idx]
                        residual, original, reconstruction = autoencoder(sample)
                        ok_test_log.append((original, reconstruction, residual))
                    logdir_ok = logdir + "_OK.pth"
                    torch.save(ok_test_log, logdir_ok)
                    
                    # Test case capacitance anomaly
                    test_cap_file = f'simulation/3RLC_Sim/data/test/3RLC_capacitance_{test_case}.csv'
                    test_cap_set = load_data(test_cap_file, window_size, step_size, voltage_scaler, current_scaler)
                    test_cap_set = TimeSeriesDataset(test_cap_set)
                    test_cap_loader = DataLoader(test_cap_set, batch_size=1, shuffle=False, num_workers=8)
                    cap_test_log = []
                    for idx in range(len(test_cap_set)):
                        sample = test_cap_set[idx]
                        residual, original, reconstruction = autoencoder(sample)
                        cap_test_log.append((original, reconstruction, residual))
                    logdir_cap = logdir + "_CAP.pth"
                    torch.save(cap_test_log, logdir_cap)
                    
                    # Test case inductance anomaly
                    test_ind_file = f'simulation/3RLC_Sim/data/test/3RLC_inductance_{test_case}.csv'
                    test_ind_set = load_data(test_ind_file, window_size, step_size, voltage_scaler, current_scaler)
                    test_ind_set = TimeSeriesDataset(test_ind_set)
                    test_ind_loader = DataLoader(test_ind_set, batch_size=1, shuffle=False, num_workers=8)
                    ind_residuals = []
                    ind_test_log = []
                    for idx in range(len(test_ind_set)):
                        sample = test_ind_set[idx]
                        residual, original, reconstruction = autoencoder(sample)
                        ind_test_log.append((original, reconstruction, residual))
                    logdir_ind = logdir + "_IND.pth"
                    torch.save(ind_test_log, logdir_ind)
                    
                    # Test case resistance anomaly
                    test_res_file = f'simulation/3RLC_Sim/data/test/3RLC_resistance_{test_case}.csv'
                    test_res_set = load_data(test_res_file, window_size, step_size, voltage_scaler, current_scaler)
                    test_res_set = TimeSeriesDataset(test_res_set)
                    test_ok_loader = DataLoader(test_res_set, batch_size=1, shuffle=False, num_workers=8)
                    res_residuals = []
                    res_test_log = []
                    for idx in range(len(test_res_set)):
                        sample = test_res_set[idx]
                        residual, original, reconstruction = autoencoder(sample)
                        res_test_log.append((original, reconstruction, residual))
                    logdir_res = logdir + "_RES.pth"
                    torch.save(res_test_log, logdir_res)
