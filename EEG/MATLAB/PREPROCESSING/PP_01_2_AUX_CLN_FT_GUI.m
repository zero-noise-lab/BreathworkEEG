
% ========================================================================
% This script loads the raw, downsampled (500hz) AUX files from DATAPATH
% - Filters ECG
% - Plots each channel, asks if it is "flipped" (peaks down)
% - Detects noisy segments (by noise in data below peak height)
% - Marks bad segments for each channel
% - Adds filtered channels to FT structure, NaN's bad segments
% - Saves new AUX-only FT structure to SAVEPATH
% Ece Kaya 2026
% ========================================================================

clear all; clc; close all;

restoredefaultpath;
addpath 'PATH_TO_FIELDTRIP';   % e.g. '/path/to/fieldtrip-20240417'
ft_defaults;

DATAPATH = 'PATH_TO_DOWNSAMPLED_DATA';
SAVEPATH = 'PATH_TO_SAVE_AUX_CLEANED';
mkdir(SAVEPATH)

files = dir(fullfile(DATAPATH,'*.mat'));

params = struct();
params.epoclen = 0.5;
params.epocoverlap = 0.5;
params.trlAmp_thresh = [0 99];
params.trlVar_thresh = [0 99];

%% LOAD - CLEAN - SAVE

for file = 1:length(files)

    datapath = fullfile(DATAPATH, files(file).name);
    fileID = extractBefore( files(file).name,'_ds_500hz.mat');

    fileSavepath = fullfile(SAVEPATH, [fileID '_aux_cln.mat']);

    if isfile(fileSavepath)
        disp(['info: ecg already cleaned for ' fileID]);
    else

        % ============= LOAD & GET ECG =============
        datasesh_ds = [];
        load(datapath);

        if numel(datasesh_ds.label)<67
            input(['no AUX channels: ' fileID]);
        else

        % pick channels
        cfg = [];
        cfg.channel = {'RESP'};
        resp_raw = ft_selectdata(cfg,datasesh_ds);

        % pick channels
        cfg = [];
        cfg.channel = {'ECG','ECG2'};
        ecg_raw = ft_selectdata(cfg,datasesh_ds);

        % ============= CLEAN ECG =============

        % ------ apply filters
        cfg = [];
        cfg.lpfilter = 'yes';
        cfg.lpfreq = 30;
        cfg.lpfiltord = 16;
        cfg.lpfilttype = 'firws';
        cfg.hpfilter = 'yes';
        cfg.hpfreq = 5;
        cfg.bsfilter = 'yes';
        cfg.bsfreq = [49 51];
        ecg_flt = ft_preprocessing(cfg,ecg_raw);

        % ==== CLEAN ===
        cfg = [];
        cfg.length   = params.epoclen;
        cfg.overlap  = params.epocoverlap;
        ecg_flt_ep = ft_redefinetrial(cfg, ecg_flt);

        nEpoc = size(ecg_flt_ep.trial,2);

        trlAmp = nan(nEpoc, 2);

        trlVar = nan(nEpoc, 2);

        for t = 1:nEpoc
            trldat = ecg_flt_ep.trial{t};

            for ch = 1:2

                X = trldat(ch,:);
                absX = abs(X);

                selX = absX < mad(absX)*3; %& absX > (median(absX)-mad(absX));
                trlAmp(t,ch) = sum(absX(selX));
                trlVar(t,ch) = nanvar(absX(selX));
            end
        end

        badepocs = isoutlier(trlAmp,"percentiles",params.trlAmp_thresh) ...
            & isoutlier(trlVar,"percentiles",params.trlVar_thresh);

        sampi_remove = struct();
        ecg_flt.chanPolInv = [0 0];

        for chan = 1:2

            % PLOT: marked segments
            cfg = [];
            cfg.blocksize = 60;
            cfg.channel = ecg_flt.label{chan};
            ft_databrowser(cfg,ecg_flt);

            % --- fix polarity
            clc;
            pol_fix_chan_Y = input('Invert channel polarity? = 1/0 :');

            if pol_fix_chan_Y == 1
                ecg_flt.trial{1,1}(chan,:) = -1.*ecg_flt.trial{1,1}(chan,:);
                ecg_flt.chanPolInv(chan,1) = 1;
            end
            close;

            if any(badepocs(:,chan))

                badEpoc_tf = badepocs(:,chan);
                % ---- use this only with overlapping epoching ----
                % get sample indices of epocs
                d = [0; diff(badEpoc_tf)];
                epIdx_starts = find(d == 1);
                epIdx_ends   = find(d == -1) - 1;

                if badEpoc_tf(1) == 1
                    epIdx_starts = [1; epIdx_starts];
                end

                if length(epIdx_ends) < length(epIdx_starts) && badEpoc_tf(end) == 1
                    epIdx_ends = [epIdx_ends; nEpoc];
                end

                sampi_remove(chan).samps = [ecg_flt_ep.sampleinfo(epIdx_starts,1) ecg_flt_ep.sampleinfo(epIdx_ends,2)];

            end

            % PLOT: marked segments
            cfg = [];
            cfg.blocksize = 60;
            cfg.channel = ecg_flt.label{chan};
            cfg.artifactalpha = 0.8;
            cfg.artfctdef.badSignal.artifact = sampi_remove(chan).samps;
            cfg = ft_databrowser(cfg,ecg_flt);

            % if there are changes
            sampi_remove(chan).samps = cfg.artfctdef.badSignal.artifact;

        end


        % --- nan out
        ecg_flt_nan = ecg_flt;
        ecg_flt_nan.label{1} = 'ECG-flt';
        ecg_flt_nan.label{2} = 'ECG2-flt';
        for chan = 1:2
            for i = 1:size(sampi_remove(chan).samps,1)
                ecg_flt_nan.trial{1,1}(chan,sampi_remove(chan).samps(i,1):sampi_remove(chan).samps(i,2)) = nan;
            end
        end

        % PLOT: marked segments
        cfg = []; cfg.blocksize = 60*5;
        ft_databrowser(cfg,ecg_flt_nan);

        % ========= MERGE ALL + INFO =========

        cfg = [];
        aux_merged = ft_appenddata(cfg, ecg_raw, ecg_flt_nan, resp_raw);

        aux_merged.filename = fileID;
        aux_merged.events = datasesh_ds.events;
        aux_merged.trl_pass = datasesh_ds.trl_pass;
        aux_merged.trl_act = datasesh_ds.trl_act;


        % PLOT: marked segments
        cfg = []; cfg.blocksize = 60*5;
        ft_databrowser(cfg,aux_merged);

        input('press key to save');
        close all;

        save(fileSavepath,'sampi_remove','aux_merged')

        end % if AUX channels present
    end
end % <<<<<<

clc;

disp(['*********** all files in ' DATAPATH ' are done! *************'])