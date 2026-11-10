function PP_01_3_AUX_PROCESS_fun(aux_fname, SAVEPATH, BB_OUTDIR)

% This script loads the filtered, noise-tagged FT AUX files from DATAPATH
% - Detects ECG PEAKS for each channel
% - Cleans PEAKS by bad intervals
% - Extracts ECG-based RESP channels (peak height interpolation)
% - Saves the cleaned & processed AUX-only FT structure to SAVEPATH
% - (signals are still NOT cut by events!)

% Usage: PP_01_3_AUX_PROCESS_fun(aux_fname, SAVEPATH, BB_OUTDIR)
% BB_OUTDIR is optional: pass '' or [] if no BBSIG derivative is available --
% subjects without BBSIG output get NaN-filled BBSIG channels instead.

addpath 'PATH_TO_PP_SCRIPTS';
addpath 'PATH_TO_FIELDTRIP';
ft_defaults;

% -------------------------------------------------------------------------
% SKIP CHECK
% -------------------------------------------------------------------------
[filepath, fname] = fileparts(aux_fname);
fileID    = extractBefore(fname, '_aux_cln');
save_path = fullfile(SAVEPATH, [fileID '_aux_processed.mat']);

if isfile(save_path)
    fprintf('Already processed: %s -- skipping\n', fileID);
    return
end

fprintf('\n========================================\n');
fprintf('Processing: %s\n', fileID);
fprintf('========================================\n');

% -------------------------------------------------------------------------
% LOAD
% -------------------------------------------------------------------------
loaded     = load(aux_fname);
aux_merged = loaded.aux_merged;

fs     = aux_merged.fsample;
nSamps = size(aux_merged.trial{1,1}, 2);

% -------------------------------------------------------------------------
% ECG PEAK DETECTION
% -------------------------------------------------------------------------
aux_merged.label{end+1} = 'PKS-1';
aux_merged.label{end+1} = 'PKS-2';

badSegMask = false(2, nSamps);

for chan = 1:2

    ecg_chan = aux_merged.trial{1,1}(chan+2, :);

    [~, locs] = fun_ecg_peakdetect(ecg_chan, fs, 1);

    % --- RR-based outlier detection
    rr    = diff(locs);
    badRR = false(size(rr));
    badRR = badRR | rr < 0.3 * fs;
    badRR = badRR | rr > 3.0 * fs;
    badRR = badRR | isoutlier(rr, 'median');

    % --- tag bad segments
    badIdx = find(badRR);
    for k = 1:numel(badIdx)
        a = locs(badIdx(k));
        b = locs(badIdx(k) + 1);
        badSegMask(chan, a:b) = true;
    end
    badSegMask(chan, isnan(ecg_chan)) = true;

    % --- write peak channel
    chanIdx = find(strcmp(aux_merged.label, ['PKS-' num2str(chan)]) == 1);
    y       = zeros(1, nSamps);
    y(locs) = ecg_chan(locs);
    aux_merged.trial{1,1}(chanIdx, :) = y;

end

% --- store bad segment mask
aux_merged.trial{1,1}(end+1, :) = badSegMask(1, :);
aux_merged.label{end+1}         = 'RR-BAD-1';
aux_merged.trial{1,1}(end+1, :) = badSegMask(2, :);
aux_merged.label{end+1}         = 'RR-BAD-2';

% -------------------------------------------------------------------------
% RESPIRATION -- ECG ENVELOPE
% -------------------------------------------------------------------------
for chan = 1:2

    chanIdx  = find(strcmp(aux_merged.label, ['PKS-' num2str(chan)]) == 1);
    pks_sig  = aux_merged.trial{1,1}(chanIdx, :);
    locs_idx = find(pks_sig > 0);
    pks_vals = pks_sig(locs_idx);

    tR = locs_idx / fs;
    t  = (0:length(pks_sig)-1) / fs;

    EDR_amp = interp1(tR, pks_vals, t);
    EDR_amp = filloutliers(EDR_amp, 'nearest');
    EDR_amp(isnan(EDR_amp)) = 0;

    aux_merged.trial{1,1}(end+1, :) = EDR_amp;
    aux_merged.label{end+1}         = ['ecg-' num2str(chan) '-envelope'];

end

% -------------------------------------------------------------------------
% BBSIG (optional -- if a subject has no BBSIG derivative, fill with NaN
% channels instead of skipping, so every subject's file has the same
% channel set)
% -------------------------------------------------------------------------
ECG_1_BB_SIG  = nan(1, nSamps);
ECG_1_BB_DATA = struct('rpeaks', struct('ECG_R_Peaks_AutoCorr', []));

tmp           = split(fileID, '-');
bb_folder_PID = fullfile(BB_OUTDIR, ['sub-' tmp{1}], 'eeg');
bb_tsv_SESH   = fullfile(bb_folder_PID, ['sub-' tmp{1} '_task-' tmp{2} '_ecg-cleaned.tsv.gz']);
bb_json_SESH  = fullfile(bb_folder_PID, ['sub-' tmp{1} '_task-' tmp{2} '_ecg-preproc.json']);

if isfolder(bb_folder_PID) && isfile(bb_tsv_SESH) && isfile(bb_json_SESH)

    % --- BBSIG cleaned signal
    gunzip(bb_tsv_SESH, tempdir);
    [~, name, ~] = fileparts(bb_tsv_SESH);
    tmp_file     = fullfile(tempdir, name);
    tmp_data     = readtable(tmp_file, 'FileType', 'delimitedtext', 'Delimiter', '\t');
    delete(tmp_file);

    ECG_1_BB_SIG = tmp_data.ecg_cleaned;

    % --- BBSIG R peaks
    fid  = fopen(bb_json_SESH, 'r');
    raw  = fread(fid, inf, 'uint8=>char')';
    fclose(fid);
    ECG_1_BB_DATA = jsondecode(raw);

else
    fprintf('  no bbsig output for %s\n', fileID);
end

% ++++ add BBSIG cln signal (NaN channel if not available)
aux_merged.trial{1,1}(end+1, :) = ECG_1_BB_SIG;
aux_merged.label{end+1}         = 'ECG-BBSIG-FLT';

% ++++ add BBSIG R peaks (NaN channel if not available)
y = nan(1, nSamps);
if ~isempty(ECG_1_BB_DATA.rpeaks.ECG_R_Peaks_AutoCorr)
    y(:) = 0;
    y(ECG_1_BB_DATA.rpeaks.ECG_R_Peaks_AutoCorr) = 1;
end
aux_merged.trial{1,1}(end+1, :) = y;
aux_merged.label{end+1}         = 'ECG-BBSIG-PKS-1';

% -------------------------------------------------------------------------
% ZSCORE VERSION
% -------------------------------------------------------------------------
aux_merged_z = aux_merged;
for ii = 1:numel(aux_merged_z.label)
    aux_merged_z.trial{1,1}(ii, :) = zscore_nan(aux_merged_z.trial{1,1}(ii, :));
end

% -------------------------------------------------------------------------
% SAVE
% -------------------------------------------------------------------------
fprintf('Saving: %s\n', save_path);
save(save_path, 'aux_merged', 'aux_merged_z', 'ECG_1_BB_DATA', '-v7.3');
fprintf('Done: %s\n', fileID);

end