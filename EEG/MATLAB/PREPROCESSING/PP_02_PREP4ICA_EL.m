
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Janina Klarmann
% Date: 27.03.2025
%
% This script loads the raw datasets, rejects bad channel and then
% performs the ICA to create datasets that contain the weights.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ========================================================================
% The adapted script reads raw resampled FT datasets from DATAPATH
% - converts them to EEGLAB (EL) format
% - drops non-EEG channels and applies standard_1020 channel lookup
% - HPF = 1 Hz, LPF = 45 Hz
% - !!! CUTS from EO beginning - EC end (session data), with 5 sec padding
%   before/after
% - Detects session-general bad channels (> mean + 3*std across channels)
% - TASK: interactively confirm + add bad channels --> they are removed
%   before ICA !!!
% - saves outlier-channel diagnostic plot (topoplot, std per channel,
%   std-over-time heatmaps) as .png
% - saves bad-channel-removed EL sets (*-chan_rmv.set) to SAVEPATH
% ========================================================================

clear all; clc; close all;

restoredefaultpath;
addpath 'PATH_TO_FIELDTRIP';
ft_defaults;
addpath 'PATH_TO_EEGLAB';
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

DATAPATH = 'PATH_TO_DOWNSAMPLED_DATA';
SAVEPATH = 'PATH_TO_SAVE_CHAN_REMOVED_DATA';
mkdir(SAVEPATH)

% ======== Hyperparameter==========
params = struct();
params.HPF = 1;
params.LPF = 45;
params.padding_sec = 5; % 5 seconds before & after the recording period (EO)
params.multip_chan_std = 3; % multiplier for threshold: mean(chan) * x std

files = dir(fullfile(DATAPATH, '*.mat'));

%% 1) LOAD - CONVERT TO EEGLAB - FILTER - CUT - SAVE RAW
clc;

for idx = 1:length(files)

    clearvars -except DATAPATH SAVEPATH params idx files

    if isfile([extractBefore([SAVEPATH filesep files(idx).name],'.mat') '-chan_rmv.set'])
        disp(['Skipping (already-processed) : ', files(idx).name])
    else

        disp(['Loading Dataset: ', files(idx).name])

        % load raw downsampled session file
        loadedS = load([files(idx).folder filesep files(idx).name]);
        raw = loadedS.datasesh_ds;

        % convert ft to eeglab (downsampled mat files)
        EEG = fieldtrip2eeglab(raw, raw.trial);

        % get session ID - name the set meaningful
        EEG.subject = extractBefore(files(idx).name,'.mat');
        EEG.setname = [EEG.subject, '-EEG_raw'];

    % ------------------------------------
    % CHANNEL INFO
    % ------------------------------------
        % drop the non-EEG channels
        EEG = pop_select(EEG,'rmchannel', 64:EEG.nbchan);

        % use standard_1020.elc in both toolboxes!
        EEG = pop_chanedit(EEG, 'lookup','standard_1020.elc');

        % check channel info
        chanlabels_system = raw.label(1:63);
        chanlabels_el = {EEG.chanlocs.labels}';
        if ~isequal(chanlabels_system,chanlabels_el)
            error('channel labeling mismatch!!!')
        end

        EEG = eeg_checkset(EEG);

    % ------------------------------------
    % FILTER
    % ------------------------------------

        EEG = pop_eegfiltnew(EEG, 'locutoff', params.HPF);

        % LPF before cutting
        EEG = pop_eegfiltnew(EEG, 'hicutoff', params.LPF);

    % ------------------------------------
    % CUT relevant data (with padding)
    % ------------------------------------

        disp([EEG.subject, '. Excluding data before & after the experiment..'])

        padded_cut_START_sample = raw.trl_pass(1,1) - params.padding_sec*raw.fsample;
        padded_cut_END_sample = raw.trl_pass(end,2) + params.padding_sec*raw.fsample;

        EEG = pop_select(EEG, 'point', [padded_cut_START_sample padded_cut_END_sample]);
        EEG = eeg_checkset(EEG);

        % --- keep the trial sample indices ----
        EEG.ft_trl_pass = raw.trl_pass;
        EEG.ft_trl_act = raw.trl_act;

    % ------------------------------------
    % MARK & REMOVE BAD CHANNELS
    % ------------------------------------

        % bad channel export
        channel_std = std(EEG.data, [],2); % calculate stand. dev. across channels
        thres = mean(mean(channel_std) + params.multip_chan_std * std(channel_std)); % thresh for bad channel marking -> higher also possible
        badchan_auto_idx = find(channel_std>thres); % store index of chan with higher std than thresh

        fig = figure;
        set(gcf,"Position",[0 0 1000 600]);

        subplot(3, 3, 1);
        topoplot(channel_std, EEG.chanlocs);
        title('Topoplot of mean std over time');

        subplot(3, 3, [2 3]); % mean std of each channel + threshold
        stem(channel_std, 'k'); hold on;
        yline(thres,'r');
        xticks([1:63]); xlim([0 64]); xtickangle(45);
        plottitle = ['outlier chans: ', mat2str(badchan_auto_idx)];
        title(plottitle, 'Interpreter', 'none');

        % plot channel_std over time (image plot): for each subject + each run
        % cut windows, calculate stds, plot in image
        sec = 10;
        LeWin = EEG.srate*sec;  % define window length
        timeVec = 0 : 1/EEG.srate : sec-1/EEG.srate; % define time vector
        idx_loop = 1:LeWin:size(EEG.data,2); % check how loop index will look like
        rms_t = zeros(EEG.nbchan, length(idx_loop)); % pre-allocate matrix of rms over time
        row_count = 1; % set counter
        for ii = 1: LeWin: size(EEG.data,2)-LeWin
            signal = EEG.data(:,ii:ii+(LeWin-1)); % get short extract from data
            rms_t(:, row_count) = std(signal, [],2); % calc std across channels
            row_count = row_count +1; % update counter
        end

        % create colormap for channel stds over time
        subplot(3, 2, [3 4]);
        ax1 = imagesc(rms_t);
        colorbar;
        colormap turbo;
        xlabel('Time (10s windows)'); ylabel('Channels');
        title(EEG.setname, 'Interpreter', 'none');

        rms_ti = rms_t > thres; % color code if value is > thresh or not
        subplot(3, 2, [5 6]);
        ax2 = imagesc(rms_ti);
        colorbar;
        xlabel('Time (10s windows)'); ylabel('Channels');
        title(['Threshold: ', num2str(thres)], 'Interpret', 'none');

        sgt = sgtitle(['Outlier channels over time for ', EEG.subject], 'Interpreter', 'none'); % title
        drawnow % print for inspection

        EEG.badchans = []; % pre-allocate bad channel index
        EEG.algoDis = []; % pre-allocate index for algorithm disagreement
        EEG.addbadchans = []; % pre-allocate additional bad channel index

        % mark + exclude bad channnels
        if ~isempty(badchan_auto_idx)
            for bc = 1:length(badchan_auto_idx)
                exl_chan = input(['Do you want to exclude channel: ', mat2str(badchan_auto_idx(bc)), '? [1/0] ']);

                if exl_chan == 1 % if agreed
                    EEG.badchans = [EEG.badchans, badchan_auto_idx(bc)]; % define bad channels
                elseif exl_chan == 0 % if disagreed
                    EEG.algoDis = [EEG.algoDis; {badchan_auto_idx(bc), datetime('today')}]; % save info to keep chan
                end
            end

            disp(['The following channels are marked as bad: ', mat2str(EEG.badchans), '.'])
            pause(2)

            addchan = input('Do you want to exclude further channels? [1/0] ');

            if addchan == 1
                excl_ac = input('Which channel(s) do you want to exclude additionally? ');
                EEG.addbadchans = excl_ac; % note down the added bad channels
                EEG.badchans = unique([EEG.badchans, excl_ac]); % ensure uniqueness
                disp(['Channel(s) ', mat2str(excl_ac), ' was / were added to the bad channels.'])
            elseif addchan == 0
                disp('Proceeding with saving all the information.')
            else
                warning('You entered neither 0 nor 1. Proceeding anyways...')
            end

        else
            % ask if further channels shall be removedif no bad channels are detected
            disp('No bad channels. ')
            pause(2)
            addchan = input('Do you want to exclude further channels? [1/0] ');

            if addchan == 1
                excl_ac = input('Which channel(s) do you want to exclude additionally? ');
                EEG.addbadchans = excl_ac; % note down the added bad channels
                EEG.badchans = unique([EEG.badchans, excl_ac]); % ensure uniqueness
                disp(['Channel(s) ', mat2str(excl_ac), ' was / were added to the bad channels.'])
            elseif addchan == 0
                disp('Proceeding with saving all the information.')
            else
                warning('You entered neither 0 nor 1. Proceeding anyways...')
            end
        end

        % externally save bad chan data
        EEG.badchans = sort(EEG.badchans);
        EEG.badchanLabels = {EEG.chanlocs(EEG.badchans).labels};

        % save the current chanlocs for later
        EEG.urchanlocs = EEG.chanlocs;

        % ------------- remove the bad channels
        EEG = pop_select(EEG, 'nochannel', EEG.badchans);

        % save the EEG.set file
        EEG.setname = [EEG.subject, '-chan_rmv'];

        EEG = pop_saveset(EEG, 'filename',[EEG.setname,'.set'],'filepath', fullfile(SAVEPATH));
        disp(['Dataset ', EEG.setname, ' saved'])

        saveas(gcf, char(fullfile(SAVEPATH, ['OutlierChans_', EEG.subject, '.png'])), 'png'); % save plot (naming not accurate - should be sub-1?)
        close;

    end
end

disp('---- all files done -----')

eeglab redraw

close all;