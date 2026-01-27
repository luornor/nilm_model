% 1) Do we have channel 2 anywhere?
d2 = dir(fullfile(pwd,'**','Samples_*.2.bin'));
d2u = dir(fullfile(pwd,'**','Samples_*_2.bin'));
disp(length(d2)); disp(length(d2u));

% 2) Show one sample's bin filenames
d = dir(fullfile(pwd,'Natural','**','Samples_003*.bin'));
{d.name}'
