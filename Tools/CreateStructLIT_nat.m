function [processed] = CreateStructLIT_nat(subset,aqID,waveform,file_offset_in_loadset)

%  CreateStructLIT
%   Author: Bruna Molinari, Douglas Renaux

%
%   Creates a struct with acquisition data in 'Out' folder
%
%   Params:
%   1] subset: specifies the dataset by typing its name as:
%   'Natural', 'Synthetic', 'Sim_Ideal', 'Sim_Induct',
%   'Sim_Induct_Harmo', 'Sim_Induct_Harmo_SNR_10',
%   'Sim_Induct_Harmo_SNR_30' or 'Sim_Induct_Harmo_SNR_60'.
%
%   2] aqID: name of the acquisition to be read
%
%   3] waveform: It's a integer number from 0 to 15. 

%% 1 - Structures with the description of loads and acquisition sequence

    acquisition_desc = [
        struct('id','7H2A2B2C2K2F2D2','name','Natural 2h',                    'path','7\7H2A2B2C2K2F2D2', 'procedure','natural');
    ];
    
    load_desc = [
        struct('id','7H2A2B2C2K2F2D2','name','Pedestal Fan, Aquarium Digital Thermostat, Aquarium Light Fish Lamp 1, Aquarium Light Fish Lamp 2, Drill, Hot-air Hand Tool, Hair Dryer - Fan 2');
     ];
 %% End of 1

fprintf("Processing: %s wf:%d\n", aqID, waveform);

%% 2 - Check for valid values of input parameters

[subset,aqID,waveform] = VerifyingVar(subset, aqID, waveform);

%% 3 - Set paths
mainPath = pwd;
cd RAW_Data
cd(subset);

% --- Natural subset has an extra folder level like "7loads_2h" ---
if strcmp(subset,'Natural')
    % Choose the first folder matching "*loads_*" (e.g., "7loads_2h")
    d = dir(fullfile(pwd, '*loads_*'));
    d = d([d.isdir]);
    if isempty(d)
        error('Natural subset folder "*loads_*" not found under Natural/. Check your folder structure.');
    end
    cd(d(1).name);  % e.g. cd('7loads_2h')
end

numLoads = extractBefore(aqID, 2);  % "7"
cd(numLoads);

if (~isfolder(aqID))
    error('Directory not found for the informed acquisition ID')
end
cd(aqID);
pathFile = pwd;



%% 4 - Processing the Description file
if(~isfile('Description.txt'))
    error("Description file does not exist");
end

load1 = '';
load2 = load1;
load3 = load1;
load4 = load1;
load5 = load1;
load6 = load1;
load7 = load1;
load8 = load1;

descriptionFileID = fopen(strcat(pathFile,"\Description.txt"),'r');
line = fgetl(descriptionFileID);
while ~feof(descriptionFileID)
    line = fgetl(descriptionFileID);
    if length(line) >= 3
        if(strcmp(string(extractBetween(line,1,3)),"Dur"))
            duration = extractAfter(line,"=");
            duration = str2double(duration);
            continue;
        end
    end
    
    if length(line) >= 6
        if(strcmp(string(extractBetween(line,1,6)),"Load 1"))
            load1 = extractAfter(line,":");
            continue;
        end
        if(strcmp(string(extractBetween(line,1,6)),"Load 2"))
            load2 = extractAfter(line,":");
            continue;
        end
        if(strcmp(string(extractBetween(line,1,6)),"Load 3"))
            load3 = extractAfter(line,":");
            continue;
        end
        if(strcmp(string(extractBetween(line,1,6)),"Load 4"))
            load4 = extractAfter(line,":");
            continue;
        end
        if(strcmp(string(extractBetween(line,1,6)),"Load 5"))
            load5 = extractAfter(line,":");
            continue;
        end
        if(strcmp(string(extractBetween(line,1,6)),"Load 6"))
            load6 = extractAfter(line,":");
            continue;
        end
        if(strcmp(string(extractBetween(line,1,6)),"Load 7"))
            load7 = extractAfter(line,":");
            continue;
        end
        if(strcmp(string(extractBetween(line,1,6)),"Load 8"))
            load8 = extractAfter(line,":");
            continue;
        end
    end
    
end
fclose(descriptionFileID);



if ~exist('duration','var')
    error("Error on reading duration from Description.txt")
end

%% Creating arrays from sample files
strBase = "\Samples_";

    strNum = pad(int2str(waveform),3,'left','0');
    
    % Verifying Files
    % If an .events file and a .config_processed file don't exist, it
    % should not there be more files to be read
    if(exist(strcat(pathFile,strBase,strNum,".events"), 'file') ~= 2)         
        fprintf("End of files reached:%s\n", strNum);
        processed = false;
        return;
    end
    
    if(exist(strcat(pathFile,strBase,strNum,".config_processed"), 'file') ~= 2)
        error("Configuration file does not exist");
    end
    
    if(exist(strcat(pathFile,strBase,strNum,"_0.bin"), 'file') ~= 2)
        error("Binary file from channel 0 does not exist")
    end
    
    if(exist(strcat(pathFile,strBase,strNum,"_1.bin"), 'file') ~= 2)
        error("Binary file from channel 1 does not exist")
    end
    
    
    %% Gains and Offset reading
    numGainsRead = 0;
    numOffsetsRead = 0;
    
    gain = zeros(3,1);
    offset = zeros(3,1);
    
    % Read the gains from the configuration file
    configFileID = fopen(strcat(pathFile,strBase,strNum,".config_processed"));
    line = fgetl(configFileID);
    
    while(line ~= -1)
        if(extractBetween(line,1,3) == "Ki=" || extractBetween(line,1,3) == "Kv=")
            line = extractAfter(line,"=");
            vls = strsplit(line,",");
            
            for index_2 = 1:length(vls)
                if(numGainsRead >= 3)
                    continue
                end
                gain(numGainsRead+1) = str2double(vls(index_2));
                numGainsRead = numGainsRead + 1;
            end
        elseif(strcmp(line,'EventGlossary='))
            line = fgetl(configFileID);
            glossaryTable = strsplit(line,',');
            line = fgetl(configFileID);
            while(line ~= -1)
                glossaryTable = [glossaryTable; strsplit(line,',')];
                line = fgetl(configFileID);
            end
        end
        line = fgetl(configFileID);
    end
    
    fclose(configFileID);
    
    
    % Read offsets from the configuration file
    configFileID = fopen(strcat(pathFile,strBase,strNum,".config_processed"));
    line = fgetl(configFileID);
    
    while(line ~= -1)
        if(length(line) < 12)
            line = fgetl(configFileID);
            continue
        end
        
        if(extractBetween(line,1,12) == "ZeroOffsetI=" || extractBetween(line,1,12) == "ZeroOffsetV=")
            line = extractAfter(line,"=");
            vls = strsplit(line,",");
            
            for index_2 = 1:length(vls)
                if(numOffsetsRead >= 3)
                    continue
                end
                offset(numOffsetsRead+1) = str2double(vls(index_2));
                numOffsetsRead = numOffsetsRead + 1;
            end
        end
        
                if (length(line) >= 14 && extractBetween(line,1,14) == "GridFrequency=")
            line = extractAfter(line,'=');
            mains_freq = str2double(line);
        end
        
        if (length(line) >= 17 && extractBetween(line,1,17) == "SamplesFrequency=")
            line = extractAfter(line,'=');
            sps = str2double(line);
        end
        
        line = fgetl(configFileID);
    end
    
    fclose(configFileID);
    
    %% Reading and converting the samples files to -> vec_0, vec_1 and vec_2
    binFileID = fopen(strcat(pathFile,strBase,strNum,"_0.bin"));
    vec = fread(binFileID, Inf, 'uint16');
    fclose(binFileID);
    
    vec_0 = ((vec ./ 2) - offset(1)) .* gain(1);
    
    binFileID = fopen(strcat(pathFile,strBase,strNum,"_1.bin"));
    vec = fread(binFileID, Inf, 'uint16');
    fclose(binFileID);
    
    vec_1 = ((vec ./ 2) - offset(2)) .* gain(2);
    
    
    %% constructing the events detection marking
    vec_detection = zeros(length(vec_1),1);
    
    eventsFileID = fopen(strcat(pathFile,strBase,strNum,".events"));
    timestampStart = str2double(fgetl(eventsFileID));
    
    line = fgetl(eventsFileID);
    while(line ~= -1)
        cells = split(line,',');
        ts = char(cells(1));
        offsetIndex = str2double(char(cells(2)));
        event = str2double(char(cells(3)));
        
        timestampEvent = posixtime(datetime(ts,'InputFormat','yyyy:MM:dd:HH:mm:ss'));
        nSeconds = timestampEvent - timestampStart;
        %comentado por erig inicio em 6/12/2020
        %sCnt = -1; 

        %for cnt = 2:length(vec_1)
         %   if(mod(vec(cnt),2) == 1)
          %      sCnt = sCnt+1;
           % end
            
            %if(sCnt == nSeconds)
             %   vec_detection(cnt+offsetIndex) = event;
              %  break;
            %end
        %end
        %comentado por erig fim 6/12/2020
        vec_detection(nSeconds*sps+offsetIndex) = event; % adicionado pro erig em 6/12/2020

        
        line = fgetl(eventsFileID);
    end
    fclose(eventsFileID);
    det = vec_detection;
    
    %% Constructing struct
    events = 1*(det~=0);
    events( (det==4)|(det==8)|(det==12)|(det==16)|...
        (det==20)|(det==24)|(det==28)|(det==32) ) = -1;
    label = string(det);
    label((det==7)|(det==4)) = glossaryTable((string(glossaryTable)=="7"),2);
    label((det==11)|(det==8)) = glossaryTable((string(glossaryTable)=="11"),2);
    label((det==15)|(det==12)) = glossaryTable((string(glossaryTable)=="15"),2);
    label((det==19)|(det==16)) = glossaryTable((string(glossaryTable)=="19"),2);
    label((det==23)|(det==20)) = glossaryTable((string(glossaryTable)=="23"),2);
    label((det==27)|(det==24)) = glossaryTable((string(glossaryTable)=="27"),2);
    label((det==31)|(det==28)) = glossaryTable((string(glossaryTable)=="31"),2);
    label((det==35)|(det==32)) = glossaryTable((string(glossaryTable)=="35"),2);
    

    aqStruct = struct('iHall',vec_0,'vGrid',vec_1,'events_r',events,'labels',label,'duration_t',duration);

%% Save struct
cd(mainPath);
if ~isfolder('Matlab_Data')
    mkdir('Matlab_Data');
    %        addpath(genpath('Matlab'))
    fprintf("Directory 'Matlab_Data' created\n");
end
cd Matlab_Data;
% save specific waveform

% specific subset file
if ( (~strcmp(subset,'Natural')) && (~strcmp(subset,'Synthetic')))
    if ~isfolder('Simulated')
        mkdir('Simulated');
    end
    cd Simulated
end

if ~isfolder(subset)
    mkdir(subset);
    s = strcat("Directory ", subset, " created\n");
    fprintf(s);
end
cd(subset)



if ~isfolder(numLoads)
    mkdir(numLoads);
    s = strcat("Directory ", numLoads, " created\n");
    fprintf(s);
end
cd(numLoads)

if ~isfolder(aqID)
    mkdir(aqID);
    s = strcat("Directory ", aqID, " created\n");
    fprintf(s);
end
cd(aqID)

%%DR
filenumber_offset = str2num(numLoads)*10000 + file_offset_in_loadset;


acquisition_descr = '';
load_descr = '';



% for the variations of Simmulated subset use:
load_descr = char(strcat(load1,load2,load3,load4,load5,load6,load7,load8));
acquisition_descr = char(sprintf("%s: %s load(s), %u seconds",subset,numLoads,int32(aqStruct.duration_t)));


fileName = strcat('Waveform',num2str(waveform+filenumber_offset));
iHall = aqStruct.iHall;
vGrid = aqStruct.vGrid;
events_r = aqStruct.events_r;
labels = aqStruct.labels;
duration_t = aqStruct.duration_t;
load_descr_short = aqID;
save(fileName,'iHall','vGrid','events_r','labels','duration_t','mains_freq','sps','acquisition_descr','load_descr_short','load_descr','waveform');
    


cd(mainPath);
processed = true;
%fprintf("Done!\n");
end
