function PP_04_2_WICA_fun(icasetname, SAVEPATH)
% Step 2 of 2: loads the component typing saved by
% PP_04_1_ICLABEL_TYPING (after any manual edits) and applies typed wICA cleaning.

restoredefaultpath;
addpath(genpath('PATH_TO_PP_SCRIPTS'));

addpath 'PATH_TO_EEGLAB';
eeglab nogui;

TYPEPATH = fullfile(SAVEPATH, 'types');

% =========================================================================
% load
% =========================================================================
[~, fname] = fileparts(icasetname);
fileID = extractBefore(fname, '-ica_wgh_cline');

type_save_fname = fullfile(TYPEPATH, [fileID '_ictypes.mat']);
wica_save_fname = fullfile(SAVEPATH, [fileID '-wica_ok.set']);

if isfile(wica_save_fname)
    fprintf('Already cleaned: %s\n', fileID);
    return
end

if ~isfile(type_save_fname)
    error('No saved typing found for %s -- run run_IC_typing first.', fileID);
end

ictype_in = load(type_save_fname);
icTypes   = ictype_in.icTypes;

% --- apply any manual overrides stored alongside the typing
if isfield(ictype_in, 'manual_overrides') && ~isempty(ictype_in.manual_overrides)
    for k = 1:numel(ictype_in.manual_overrides)
        ov = ictype_in.manual_overrides(k);
        icTypes{ov.ic} = ov.to;
    end
end

% --- Load ICA file
EEG_ica = pop_loadset(icasetname);

% =========================================================================
% WICA
% =========================================================================
fprintf('Subject %s: %d brain / %d eye / %d muscle / %d ambiguous\n', fileID, ...
    sum(strcmp(icTypes,'brain')), sum(strcmp(icTypes,'eye')), ...
    sum(strcmp(icTypes,'muscle')), sum(strcmp(icTypes,'ambiguous')));

% --- Apply typed wICA
[cleaned_data, wicaDiag] = fun_wICA_typed(EEG_ica, icTypes);

EEG          = EEG_ica;
EEG.data     = cleaned_data;
EEG.setname  = [EEG_ica.subject, '-wica_ok'];
EEG.wicaDiag = wicaDiag;
EEG = pop_saveset(EEG, 'filename', [EEG.setname '.set'], 'filepath', SAVEPATH);
disp(['Dataset ' EEG.setname ' saved']);

end