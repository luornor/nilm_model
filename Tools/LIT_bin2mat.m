%LIT_bin2mat.m 
%   This MATLAB script is used to extract binary data from the LIT-Dataset
%   RAW_Data files and construct MATLAB arrays from this data for further
%   processing. MATLAB arrays are saved in .mat files.
%Author: Bruna Molinari, Douglas Renaux
%Documentation:
%   read the LIT_Dataset - MATLAB user guide (pdf file)

% Clear the workspace and command window
clc
clear
close all

% 1 - Select a subset:
% Type = char array
% Options =
%                    º 'Natural'
%                    º 'Synthetic'
%                    º 'Sim_Ideal'
%                    º 'Sim_Induct'
%                    º 'Sim_Induct_Harmo'
%                    º 'Sim_Induct_Harmo_SNR_10'
%                    º 'Sim_Induct_Harmo_SNR_30'
%                    º 'Sim_Induct_Harmo_SNR_60'
%                    º 'Simulated'   (NEW: runs all simulated variants)
%
subset = 'Simulated';   % user-friendly option

% Expand "Simulated" into all simulated variants
if strcmp(subset,'Simulated')
    subset_list = { ...
        'Sim_Ideal', ...
        'Sim_Induct', ...
        'Sim_Induct_Harmo', ...
        'Sim_Induct_Harmo_SNR_10', ...
        'Sim_Induct_Harmo_SNR_30', ...
        'Sim_Induct_Harmo_SNR_60' ...
    };
else
    subset_list = {subset};
end

% === Set dataset root explicitly (edit this path) ===
DATASET_ROOT = 'C:\Users\ASUS\Desktop\Projects\ML Project\Dataset';  % contains RAW_Data/ and Tools/
cd(DATASET_ROOT);
addpath(genpath(DATASET_ROOT));

% 2..5 - Run conversion for each selected subset
for s = 1:length(subset_list)

    subset = subset_list{s};
    fprintf("\n=== Processing subset: %s ===\n", subset);

    % Build list of loads for this subset
    loads_set = ChoiceOfLoads(subset);

    % Number of loads to change the files
    L = size(loads_set, 1);        % safer than length() for char matrices
    file_offset_in_loadset = 0;
    current_numLoads = "1";

    % Loop for all loads in load set
    for n = 1:L

        aqID = loads_set(n,:);

        % Determines the amount of traces (angle variation) in each acquisition.
        [n_t, numLoads] = NumberOfTraces(subset, aqID);

        if ~strcmp(current_numLoads, numLoads)
            current_numLoads = numLoads;
            file_offset_in_loadset = 0;
        end

        for trace = n_t
            % Create one .mat for each database file
            if strcmp(subset,'Natural')
                if (false == CreateStructLIT_nat(subset, aqID, trace, file_offset_in_loadset))
                    break;
                end
            else
                CreateStructLIT(subset, aqID, trace, file_offset_in_loadset);
            end
        end

        file_offset_in_loadset = file_offset_in_loadset + length(n_t);
    end
end
