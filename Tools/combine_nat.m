function [last_file] = combine_nat(first,aqId)
% sample call: n = combine_nat(70000,'7H2A2B2C2K2F2D2');
% This function combines several segments of a single waveform of the
% Natural subset into a single file.
% It requires significant amount of memory !
% The file has three vectors
% iHall - current samples
% vGrid - voltage samples
% events_r - indication of load events (+1 = ON; -1 = OFF)
% parameters:
% first - index of first matlab file with a segment of the waveform
% aqID  - name of the folder in Matlab_Data\Natural were the waveform
%         segments are stored.

 file_idx=first;
 %mainPath = pwd;
 r_dir = fileparts(mfilename('fullpath'));
 cd (r_dir);
 cd ../Matlab_Data/Natural;
 cd (aqId(1));
 cd (aqId);
 pathFile = pwd;
 fileName = strcat('Waveform',num2str(file_idx),'.mat');
 I = [];
 V = [];
 ev = [];
 lab = "";
 fprintf("Segments: ");
 while (exist(strcat(pathFile,"\",fileName), 'file') == 2)
    fprintf("%s\n ", fileName);
    load(strcat(pathFile,"\",fileName));
    I = [I; iHall];
    V = [V; vGrid];
    ev = [ev; events_r];
    %lab1 = strcat(lab,labels);
    %lab = lab1;
    file_idx = file_idx+1;
    fileName = strcat('Waveform',num2str(file_idx),'.mat');
 end
 last_file= file_idx-1;
 iHall = I;
 vGrid = V;
 events_r = ev;
 fprintf("\nSaving...\n ");
 save('Waveform','iHall','vGrid','events_r');
end
