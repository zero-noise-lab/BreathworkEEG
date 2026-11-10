function PP_04_1_ICLABEL_TYPING_fun(icasetname, SAVEPATH)
% RUN_IC_TYPING  Step 1 of 2: ICLabel + custom component typing, saved to
% disk for review before cleaning. Edit the saved typing .mat (icTypes,
% or add entries to manual_overrides) before running run_wICA_apply, if
% you want to override anything.
%
% manual_overrides format: struct array with fields ic, from, to, e.g.
%   ictype_out.manual_overrides(1) = struct('ic',7,'from','ambiguous','to','brain');

restoredefaultpath;
addpath(genpath('PATH_TO_PP_SCRIPTS'));

addpath 'PATH_TO_FIELDTRIP';
ft_defaults;

addpath 'PATH_TO_EEGLAB';
eeglab nogui;

addpath(genpath(fullfile(fileparts(which('eeglab')), 'plugins', 'Cleanline2.1')));

TYPEPATH = fullfile(SAVEPATH, 'types');

RAWPATH = 'PATH_TO_DOWNSAMPLED_DATA';

% =========================================================================
% load
% =========================================================================
[~, fname]   = fileparts(icasetname);
fileID = extractBefore(fname, '-ica_wgh_cline');

type_save_fname = fullfile(TYPEPATH, [fileID '_ictypes.mat']);

if isfile(type_save_fname)
    fprintf('Already typed: %s\n', fileID);
    return
end

% --- Load ICA file
EEG_ica = pop_loadset(icasetname);
EEG_ica = pop_chanedit(EEG_ica, 'lookup', 'standard_1020.elc');
EEG_ica.icaact = eeg_getdatact(EEG_ica, 'component', 1:size(EEG_ica.icaweights,1));
n_comps = size(EEG_ica.icaact, 1);

fprintf('\n========================================\n');
fprintf('Subject: %s\n', EEG_ica.subject);
fprintf('========================================\n');

% --- ICLabel
EEG_ica = pop_iclabel(EEG_ica, 'default');

% --- Load AUX (true EOG-cheek + ECG) + treat like EEG signal
params.HPF = 1; params.LPF = 45; params.padding_sec = 5;
loadedS = load(fullfile(RAWPATH, [EEG_ica.subject '.mat']));
raw     = loadedS.datasesh_ds;
AUX     = fieldtrip2eeglab(raw, raw.trial);

aux_keep      = {'EMG','ECG','ECG2'};  % <-- !!! EMG because our EOG was mislabeled as EMG!!!
aux_available = aux_keep(ismember(aux_keep, {AUX.chanlocs.labels}));
if isempty(aux_available)
    error('No AUX channels found for subject %s', EEG_ica.subject);
end
AUX = pop_select(AUX, 'channel', aux_available);
AUX = pop_eegfiltnew(AUX, 'locutoff', params.HPF);
AUX = pop_eegfiltnew(AUX, 'hicutoff', params.LPF);

padded_cut_START_sample = raw.trl_pass(1,1)   - params.padding_sec * raw.fsample;
padded_cut_END_sample   = raw.trl_pass(end,2) + params.padding_sec * raw.fsample;
AUX = pop_select(AUX, 'point', [padded_cut_START_sample padded_cut_END_sample]);
AUX = pop_cleanline(AUX, 'LineFrequencies', 50, 'Bandwidth', 5);
AUX = eeg_checkset(AUX);

% --- Run the typing function
[icTypes, typeDiag] = get_component_types(EEG_ica, AUX, struct());

% --- Print current typing for review
fprintf('\n%-6s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n', ...
    'IC','Type','EyeP','MuscP','BrainP','HeartP','BlinkSc','Kurt','FrontR','EyeVts','CardSc','CardCV','Slope');
fprintf('%s\n', repmat('-',1,118));
for ic = 1:n_comps
    fprintf('%-6d %-10s %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8.2f %-8d %-8.2f %-8.2f %-8.2f\n', ...
        ic, icTypes{ic}, typeDiag.eye_prob(ic), typeDiag.muscle_prob(ic), ...
        typeDiag.brain_prob(ic), typeDiag.heart_prob(ic), typeDiag.blink_score(ic), typeDiag.ic_kurtosis_blink(ic), ...
        typeDiag.frontal_topo_ratio(ic), typeDiag.eye_votes(ic), typeDiag.cardiac_score(ic), typeDiag.cardiac_cv(ic), typeDiag.muscle_slope(ic));
end
fprintf('(blink events: %d, R-peaks: %d; EOG whole-window corr available in typeDiag.eog_corr for reference)\n', ...
    typeDiag.blink_n_events, typeDiag.r_peak_n_events);

fprintf('\nFinal typing:\n');
for ic = 1:n_comps
    fprintf('  IC %02d: %s\n', ic, icTypes{ic});
end

% --- Save typing (not the cleaned data -- run_wICA_apply does that)
ictype_out                  = struct();
ictype_out.subject          = EEG_ica.subject;
ictype_out.icTypes          = icTypes;
ictype_out.typeDiag         = typeDiag;
ictype_out.manual_overrides = [];
save(type_save_fname, '-struct', 'ictype_out');
fprintf('Saved typing: %s\n', type_save_fname);

end