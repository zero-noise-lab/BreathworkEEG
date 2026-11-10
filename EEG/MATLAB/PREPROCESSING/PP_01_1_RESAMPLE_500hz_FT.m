
% ========================================================================
% This script loads the raw, high sampled (2500hz) vhdr files from DATAPATH
% - downsamples eeg data to 500 hz
% - readjusts event timings given the scaling
% - finds event markers of passive and active trials
% - saves output as FT structure, with event markers to SAVEPATH
% Ece Kaya 2026
% ========================================================================

clear all; clc; close all;

restoredefaultpath;
addpath 'PATH_TO_FIELDTRIP';   % e.g. '/path/to/fieldtrip-20240417'
ft_defaults

DATAPATH = "PATH_TO_RAW_VHDR_FOLDER/";
SAVEPATH = "PATH_TO_SAVE_DOWNSAMPLED_DATA/";
mkdir(SAVEPATH)

files = dir(fullfile(DATAPATH, '*.vhdr'));

%% resample - save

params = struct();
params.resamplefs = 500;

for file = 1:length(files)

    datapath = strcat(DATAPATH, files(file).name);
    fileID = extractBefore(files(file).name, '.vhdr');

    if isfile(strcat(SAVEPATH, fileID, '_ds_500hz.mat'))
        disp('already downsampled');
        continue
    end

    cfg          = [];
    cfg.dataset  = char(datapath);
    datasesh_raw = ft_preprocessing(cfg);

    % organize events and trials
    event_trig = ft_read_event(char(datapath));
    event_trig = event_trig(strcmp({event_trig.type}, 'Stimulus'));

    cfg = [];
    cfg.resamplefs = params.resamplefs;
    cfg.detrend    = 'no';
    datasesh_ds = ft_resampledata(cfg, datasesh_raw);

    oldFs = datasesh_raw.fsample;
    newFs = params.resamplefs;
    scale = newFs / oldFs;

    event_trig_new = event_trig;
    for i = 1:numel(event_trig_new)
        event_trig_new(i).sample = round(event_trig_new(i).sample * scale);
    end

    datasesh_ds.events = event_trig_new;

    trl = [];
    % 1st col: beginning sample
    % 2nd col: end sample
    % 3rd col: "offset"
    for e = 1:length(event_trig_new)-1
        trl(e,1) = event_trig_new(e).sample;
        trl(e,2) = event_trig_new(e+1).sample - 1;
    end
    trl(length(event_trig_new),1) = event_trig_new(end).sample;
    trl(length(event_trig_new),2) = size(datasesh_ds.trial{1,1},2);
    trl(:,3) = zeros(length(event_trig_new),1);

    % find trials of interest (passive ~4min, active ~13min)
    trl_dur_sec = diff(trl(:,1)) ./ newFs;
    pass_trls = trl_dur_sec > 60*4 & trl_dur_sec < 60*4.1;
    act_trls  = trl_dur_sec > 60*13 & trl_dur_sec < 60*13.1;

    % cut PASS trials to 4min exactly, ACT trials to 13min exactly, after event onset
    trl_pass = [trl(pass_trls,1)+1, trl(pass_trls,1)+4*60*newFs, trl(pass_trls,3)];
    trl_act  = [trl(act_trls,1)+1,  trl(act_trls,1)+13*60*newFs, trl(act_trls,3)];

    if size(trl_pass,1) < 7 || size(trl_act,1) < 4
        error('something wrong')
    end

    datasesh_ds.trl_pass = trl_pass;
    datasesh_ds.trl_act  = trl_act;

    save(strcat(SAVEPATH, fileID, '_ds_500hz.mat'), "datasesh_ds")

end

disp('---- all files done -----')


