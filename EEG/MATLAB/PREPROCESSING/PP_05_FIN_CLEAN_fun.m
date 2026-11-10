function PP_05_FIN_CLEAN_fun (ica_cln_fname, SAVEPATH)

% ================== FINAL PASSIVE-SEGMENT PIPELINE (FieldTrip) ==================
% Loads a subject's wICA-cleaned EEG dataset and checks that its channel labels/order match the expected 63-channel reference before doing anything else.
% Cuts the continuous recording into the 7 passive trials (EO_pre, EC_pre, R1, R2, R3, EC_post, EO_post), adjusting for the padding added earlier at PP_02.
% Checks each trial for bad channels (based on whole-trial signal variability), then interpolates any channel flagged as bad in ANY trial + any BAD from before ICA.
% Re-references all channels to the average.
% Saves the cleaned, trial-cut data along with a separate diagnostics file 
% ++ Fix 4 Aug 2026:
% Corrected a bug where FieldTrip's channel-repair step silently reordered channels (moved repaired/missing channels to the end of the list) instead of keeping the expected fixed order -- now explicitly reindexed back to match the canonical 63-channel order before saving.
% Removed the custom trial-labels field that had been attached to the FieldTrip structure (it wasn't needed downstream and previously caused separate structural issues).

addpath 'PATH_TO_PP_SCRIPTS';

addpath 'PATH_TO_FIELDTRIP';
ft_defaults;

addpath 'PATH_TO_EEGLAB';
eeglab nogui;

load('reset_elec_63.mat')   % elec_63, neighbours_63

% -------------------------------------------------------------------------
% PARAMS
% -------------------------------------------------------------------------
params = struct();
params.padding_sec        = 5; % !!!!!!WAS SET AT PP_02!!!!!
params.badchan_fun_std     = 3;
params.badchan_center_fun  = 'mean'; 
params.trial_labels        = {'EO_pre','EC_pre','R1','R2','R3','EC_post','EO_post'};
params.interp_method       = 'spline';       % matches EEGLAB's spherical-spline default (pop_interp), for comparability with the ASR pipeline


% -------------------------------------------------------------------------
% LOAD
% -------------------------------------------------------------------------
[filepath, filename, ext] = fileparts(ica_cln_fname);
EEG_wicad = pop_loadset('filename', [filename ext], 'filepath', filepath);
EEG_wicad = eeg_checkset(EEG_wicad);
EEG_wicad = pop_chanedit(EEG_wicad, 'lookup', 'standard_1020.elc');  % ensure channel positions present for neighbour definition

fileID       = extractBefore(filename, '-wica_ok');
fileSAVEPATH = fullfile(SAVEPATH, [fileID '_final_passive.mat']);
diagSAVEPATH = fullfile(SAVEPATH, [fileID '_final_passive_diag.mat']);

fprintf('\n========================================\n');
fprintf('Subject: %s\n', EEG_wicad.subject);
fprintf('========================================\n');

if isfile(fileSAVEPATH)
    fprintf('Already processed: %s -- skipping\n', EEG_wicad.subject);
    return
end


% --- VERIFY: EEGLAB's original channel labels match elec_63 exactly (same channels, same order)
orig_labels = upper({EEG_wicad.urchanlocs.labels})';

if isequal(orig_labels, elec_63.label)
    fprintf('Label check OK: EEG_wicad chanlocs == elec_63.label (identical order).\n');
elseif isequal(sort(orig_labels), sort(elec_63.label))
    fprintf('WARNING: same 63 channels present, but ORDER differs between EEG_wicad and elec_63!\n');
    fprintf('First mismatch at position %d: EEG_wicad=%s, elec_63=%s\n', ...
        find(~strcmp(orig_labels, elec_63.label), 1), ...
        orig_labels{find(~strcmp(orig_labels, elec_63.label), 1)}, ...
        elec_63.label{find(~strcmp(orig_labels, elec_63.label), 1)});
else
    missing_from_elec = setdiff(orig_labels, elec_63.label);
    missing_from_eeg   = setdiff(elec_63.label, orig_labels);
    fprintf('ERROR: channel SETS differ, not just order!\n');
    fprintf('In EEG_wicad but not elec_63: %s\n', strjoin(missing_from_elec, ', '));
    fprintf('In elec_63 but not EEG_wicad: %s\n', strjoin(missing_from_eeg, ', '));
end

% -------------------------------------------------------------------------
% ADJUST TRIAL INDICES  (the one unavoidable index-offset step, done once)
% -------------------------------------------------------------------------
padded_cut_START_sample = EEG_wicad.ft_trl_pass(1,1) - params.padding_sec * EEG_wicad.srate;
trl_pass_adjusted        = EEG_wicad.ft_trl_pass - padded_cut_START_sample + 1;
trl_pass_adjusted(:,3)   = 0;

nSamples = size(EEG_wicad.data, 2);
n_trials = size(trl_pass_adjusted, 1);
assert(n_trials == numel(params.trial_labels), 'ft_trl_pass has %d rows but %d trial labels given.', n_trials, numel(params.trial_labels));

trial_available = true(n_trials, 1);
for t = 1:n_trials
    if trl_pass_adjusted(t,1) < 1 || trl_pass_adjusted(t,2) > nSamples
        trial_available(t) = false;
        warning('Trial %s falls outside available data [1 %d] (adjusted range [%d %d]) -- will be excluded.', ...
            params.trial_labels{t}, nSamples, trl_pass_adjusted(t,1), trl_pass_adjusted(t,2));
    end
end
if ~all(trial_available)
    fprintf('*** %d/%d trials unavailable for %s: %s ***\n', sum(~trial_available), n_trials, ...
        EEG_wicad.subject, strjoin(params.trial_labels(~trial_available), ', '));
    error('cutting problem!')
end

% -------------------------------------------------------------------------
% CONVERT TO FIELDTRIP & CUT IMMEDIATELY
% -------------------------------------------------------------------------
data_cont           = eeglab2fieldtrip(EEG_wicad, 'raw', 'none');
data_cont.trialinfo = [];

% --- VERIFY: labels preserved correctly through eeglab2fieldtrip conversion
orig_labels = {EEG_wicad.chanlocs.labels};
assert(isequal(data_cont.label, orig_labels), ...
    'eeglab2fieldtrip changed channel labels/order at conversion! orig vs converted mismatch.');
fprintf('Label check OK at conversion: %d channels, order preserved.\n', numel(orig_labels));

cfg_ft     = [];
cfg_ft.trl = trl_pass_adjusted(trial_available, :);
data_pass  = ft_redefinetrial(cfg_ft, data_cont);

% -------------------------------------------------------------------------
% BAD CHANNEL DETECTION (whole-trial STD, must be bad in ALL available trials)
% -------------------------------------------------------------------------
n_chan_data = numel(data_pass.label);
flag_mat    = false(numel(data_pass.trial), n_chan_data);

fprintf('\n--- Per-trial channel quality check (whole-trial STD, no windowing) ---\n');
for t = 1:numel(data_pass.trial)
    [bad_idx, ~, thresh] = check_channel_quality_wholetrial(data_pass.trial{t}, params.badchan_fun_std, params.badchan_center_fun);
    flag_mat(t, bad_idx) = true;
    if isempty(bad_idx)
        fprintf(' no channels flagged (threshold=%.2f)\n', thresh);
    else
        fprintf(' %s (threshold=%.2f)\n', strjoin(data_pass.label(bad_idx)', ', '), thresh);
    end
end

flag_final = any(flag_mat);

badchanLabels_afterICA = data_pass.label(flag_final)';

% -------------------------------------------------------------------------
% INTERPOLATE (ft_channelrepair, spline method to match EEGLAB's spherical spline)
% !!!!! FT is picky about this. Make labels all UPPERCASE!!!!
% -------------------------------------------------------------------------

badchanLabels_beforeICA = EEG_wicad.badchanLabels;

badchanLabels_beforeICA = upper(badchanLabels_beforeICA);
badchanLabels_afterICA = upper(badchanLabels_afterICA);

data_pass.label = upper(data_pass.label);


if isempty(badchanLabels_afterICA) && isempty(badchanLabels_beforeICA)
    fprintf('No channels to interpolate.\n');
    data_interp = data_pass;
else
    cfg_repair            = [];
    cfg_repair.method     = params.interp_method;
    cfg_repair.neighbours = neighbours_63;
    cfg_repair.elec       = elec_63;
    cfg_repair.badchannel = badchanLabels_afterICA;

    cfg_repair.missingchannel = badchanLabels_beforeICA;

    data_interp           = ft_channelrepair(cfg_repair, data_pass);
end

% -------------------------------------------------------------------------
% RE-REFERENCE (average)
% -------------------------------------------------------------------------
cfg_reref            = [];
cfg_reref.reref      = 'yes';
cfg_reref.refchannel = 'all';
cfg_reref.channel    = 'all';
data_final           = ft_preprocessing(cfg_reref, data_interp);

fprintf('Re-referenced to average, %d channels, %d trials.\n', numel(data_final.label), numel(data_final.trial));


% !!!!!!!! fix fieldtrip mistake in channel ordering !!!!!!!!
[found, reorder_idx] = ismember(elec_63.label, data_final.label);

assert(all(found), 'Not every elec_63 channel was found in data_final.label -- stop and check before reindexing.');

data_final.label = data_final.label(reorder_idx);
for tr = 1:numel(data_final.trial)
    data_final.trial{tr} = data_final.trial{tr}(reorder_idx, :);
end

assert(isequal(data_final.label, elec_63.label), 'Reindex failed -- channel order still does not match elec_63.');


% -------------------------------------------------------------------------
% DIAGNOSTICS
% -------------------------------------------------------------------------
diagOut = struct();
diagOut.subject             = EEG_wicad.subject;
diagOut.trial_labels_all    = params.trial_labels;
diagOut.trial_available     = trial_available;
diagOut.flag_mat            = flag_mat;
diagOut.chanlabels_beforeICA = {EEG_wicad.chanlocs.labels};
diagOut.badchanLabels_beforeICA = badchanLabels_beforeICA;
diagOut.badchanLabels_afterICA = badchanLabels_afterICA;
diagOut.params              = params;
diagOut.padded_cut_START_sample = padded_cut_START_sample;
save(diagSAVEPATH, 'diagOut');
fprintf('Diagnostics saved: %s\n', diagSAVEPATH);

% -------------------------------------------------------------------------
% SAVE FINAL DATA
% -------------------------------------------------------------------------
tmpstr = split(EEG_wicad.subject, {'_','-'});

DATA               = struct();
DATA.rootID        = EEG_wicad.subject;
DATA.PID           = tmpstr{1};
DATA.TYPE          = tmpstr{2};
DATA.data_pass     = data_final;
DATA.info.badchans_interpolated = {badchanLabels_beforeICA badchanLabels_afterICA};
DATA.info.clean_params          = params;

fprintf('Saving: %s\n', fileSAVEPATH);
save(fileSAVEPATH, 'DATA', '-v7.3');
fprintf('Done: %s\n', EEG_wicad.subject);

end