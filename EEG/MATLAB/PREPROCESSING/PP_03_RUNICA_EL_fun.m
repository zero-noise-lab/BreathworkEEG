% ========================================================================
% This function runs ICA on preprocessed EL sets, adaptation from Janina
% Klarmann's script
% - loads ICA-prepped EL set (PP_02) from filename
% - removes line noise (pop_cleanline) --> this takes long
% - Componens + datasets saved at SAVEPATH
% - TASK: CALL THIS ON .sh file
% ========================================================================
% ! update 23.03.26
% - bad channels already removed from -chan-rmv set @PP_02
% - channellocations were set to standard 1020 in chan-rmv set @PP_02
% - LPF = 45hz already done @PP_02
% - loads chan-rmv set directly (we don't save raw anymore)
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function PP_03_RUNICA_EL_fun(filename, SAVEPATH)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Janina Klarmann
% Date: 26.05.2025
%
% This script performs the ICA to create datasets that contains the weights.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [datapath_root, filename_root] = fileparts(filename);
    fileID = extractBefore(filename_root,'-chan_rmv');

    setname_out = [fileID '-ica_wgh_cline'];

    if isfile(fullfile(SAVEPATH, [setname_out '.set']))
        fprintf('Skipping file: %s\n', filename);
    else

    % Load eeglab
    addpath 'PATH_TO_EEGLAB'
    eeglab;

    % --- Check if pop_cleanline exists, install if not ---
    if ~exist('pop_cleanline', 'file')
        disp('Installing CleanLine plugin...');
        eeglab; close;  % ensure EEGLAB paths are loaded
        plugin_askinstall('cleanline', 'cleanline', 1);  % silent install
    end

    % new version has standard1020 layout
    EEG = pop_loadset('filename', [fileID '-chan_rmv.set'], 'filepath', datapath_root);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Perform ICA, get weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    EEG.setname = setname_out;

    % remove line noise (this takes a bit of time)
    EEG = pop_cleanline(EEG, 'LineFrequencies', 50, 'Bandwidth', 5, 'SignalType', 'Channels');

    % do the ICA
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1);

    % store the dataset
    EEG = pop_saveset(EEG, 'filename',[EEG.setname,'.set'],'filepath', char(fullfile(SAVEPATH)));

    fprintf('Finished file: %s\n', filename);

    end
end