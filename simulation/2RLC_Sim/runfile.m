clear; % Clear workspace
anomalies = false;  % Set to true to simulate with anomalies
test = true;
variables;  % Run script to initialize all model variables

if anomalies == true  % This must run after the "variables" script
    % Specify the directory path
    anomalyPath = './anomalies';
    anomalyFiles = dir(anomalyPath);
    for f = 1:length(anomalyFiles)
        if anomalyFiles(f).isdir == 0 % Assure it's not a directory
            filename = anomalyFiles(f).name;
            disp(['Processing file: ' filename]);
            variables;  % Run script to initialize all model variables
            runtime = anomaly_runtime;  %
            scriptPath = fullfile(anomalyPath, filename);
            run(scriptPath);
            for i = 0:2:20
                % Update the value of sine_offset
                sine_offset.Value = i;
                % Call the function with updated sine_offset
                clean_name = erase(filename, '.m');
                csvname = sprintf('./data/test/2RLC_%s_%d.csv', clean_name, i);
                processFile(csvname, anomalies, test)
            end
        end
    end
else
    if test == true
        runtime = anomaly_runtime;
        for i = 0:2:20
            % Update the value of sine_offset
            sine_offset.Value = i;
            % Call the function with updated sine_offset
            csvname = sprintf('./data/test/2RLC_Test_%d.csv', i);
            processFile(csvname, anomalies, test)
        end
        
    else
        for i = 0:2:20
            sine_offset.Value = i;
            csvname = sprintf('./data/train/2RLC_%d.csv', i);
            processFile(csvname, anomalies, test)
        end
    end
end


% Function to process each file
function processFile(csvname, anomalies, test)
    simOut = sim('x2RLC.slx');
    voltage = squeeze(simOut.voltage);
    current = squeeze(simOut.current);
    
    voltage = squeeze(voltage);
    current = squeeze(current);

    T = table(voltage, ...
        current);

    if anomalies == true
        T(1:100,:) = []; % remove first 1s sample time
    else
        if test == true
            T(1:100,:) = []; % remove first 1s sample time
        else
            T(1:100100,:) = []; % remove first 1001s sample time
        end
    end

    writetable(T, csvname)
    
end