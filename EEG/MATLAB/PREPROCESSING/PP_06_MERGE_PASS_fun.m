function PP_06_MERGE_PASS_fun(eeg_fname, FINALSAVEPATH, RAWEEGPATH, AUXPATH, SURVEYPATH, SMTPATH)
% PP_06_MERGE_PASS_fun - Merge EEG, AUX, Survey and (optionally) SMT data
% Usage: PP_06_MERGE_PASS_fun(eeg_fname, FINALSAVEPATH, RAWEEGPATH, AUXPATH, SURVEYPATH, SMTPATH)
% SMTPATH is optional: pass '' or [] (or omit it) to skip SMT entirely --
% S.smt_added will just be false for every subject.
% ----
% RAWEEGPATH = actual eeg files with date info on them
% AUXPATH = processed aux files 
% SURVEYPATH = path to the survey table, loaded from .mat file 

if nargin < 6
    SMTPATH = '';
end

addpath 'PATH_TO_FIELDTRIP';
ft_defaults;

addpath 'PATH_TO_RESET_EEG_SCRIPTS';

% -------------------------------------------------------------------------
% LOAD SHARED TABLES (same for all subjects)
% -------------------------------------------------------------------------
tmp      = load(SURVEYPATH);
fn       = fieldnames(tmp);
SUR_WIDE = tmp.(fn{1});

have_smt = ~isempty(SMTPATH);
if have_smt
    tmp     = load(SMTPATH);
    fn      = fieldnames(tmp);
    SMT_TBL = tmp.(fn{2});
end

% -------------------------------------------------------------------------
% LOAD RAW EEG FILE LIST
% -------------------------------------------------------------------------
raweeg_files = dir(fullfile(RAWEEGPATH, '*.eeg'));
% ------------ get raw data path, read date, fix (EEG computer is 333 days off)

for i = 1:length(raweeg_files)
    raweeg_files(i).correct_date = datetime(raweeg_files(i).date, ...
        'InputFormat', 'dd-MMM-yyyy HH:mm:ss') + days(333);
    % correct for leap year I guess? 
    % some date mismatch starting 9 March 2026 (manually found out)
    if raweeg_files(i).correct_date > datetime('9-Mar-2026')
        raweeg_files(i).correct_date = raweeg_files(i).correct_date + days(6);
    end
    raweeg_files(i).pid = extractBefore(raweeg_files(i).name,'.eeg');
end

% -------------------------------------------------------------------------
% SKIP CHECK  (input files are *_final_passive.mat, produced by
% fun_PP_05_FIN_CLEAN -- passive trials only, already cut/cleaned/reref'd)
% -------------------------------------------------------------------------
[~, fname] = fileparts(eeg_fname);
session_ID = extractBefore(fname, '_final_passive');
save_path  = fullfile(FINALSAVEPATH, [session_ID '_passive_merged.mat']);

if isfile(save_path)
    fprintf('Skipping %s -- already merged.\n', session_ID);
    return
end

% -------------------------------------------------------------------------
% LOAD EEG (passive trials only -- no data_act anymore)
% -------------------------------------------------------------------------
fprintf('\nProcessing: %s\n', session_ID);
tmp  = load(eeg_fname);
DATA = tmp.DATA;

S            = struct();
S.rootID     = DATA.rootID;
S.PID        = DATA.PID;
S.TYPE       = DATA.TYPE;
S.PID_nr     = nan;
S.data_pass  = DATA.data_pass;
S.info       = DATA.info;

% Sanity check: this pipeline hard-errors upstream (fun_PP_05_FIN_CLEAN) if
% any trial is missing, so every merged file should have exactly 7 passive
% trials. Catch it here too in case an older/different-vintage file slips in.
if numel(S.data_pass.trial) ~= 7
    error('Expected 7 passive trials but found %d for %s -- check which pipeline version produced this file.', ...
        numel(S.data_pass.trial), session_ID);
end

if any(regexp(S.TYPE, 'BW'))
    S.sesh_type    = categorical(cellstr('BW'));
    S.sesh_bw_true = true;
elseif any(regexp(S.TYPE, 'PMR'))
    S.sesh_type    = categorical(cellstr('PMR'));
    S.sesh_bw_true = false;
else
    error('Session type unknown for %s', session_ID);
end

% -------------------------------------------------------------------------
% AUX  (passive trials only -- act cutting/appending removed)
% -------------------------------------------------------------------------
session_ID_aux = extractBefore(S.rootID, '_ds_500hz');
aux_path       = fullfile(AUXPATH, [session_ID_aux '_aux_processed.mat']);

if isfile(aux_path)
    fprintf('Adding AUX: %s\n', session_ID_aux);
    AUX_STRUCT = load(aux_path);
    aux_merged = AUX_STRUCT.aux_merged;

    cfg      = []; cfg.trl = aux_merged.trl_pass;
    aux_pass = ft_redefinetrial(cfg, aux_merged);

    % Trial count/order must match between EEG passive data and AUX passive
    % data for ft_appenddata to align them correctly -- it matches by trial
    % position, not by label, so a mismatch here would silently misalign
    % AUX channels against the wrong EEG trial.
    if numel(aux_pass.trial) ~= numel(S.data_pass.trial)
        error('AUX has %d passive trials but EEG has %d for %s -- trial counts must match before appending.', ...
            numel(aux_pass.trial), numel(S.data_pass.trial), session_ID);
    end

    S.data_pass = ft_appenddata([], S.data_pass, aux_pass);

    S.aux_added = true;
else
    fprintf('AUX not found for %s\n', session_ID_aux);
    S.aux_added = false;
end

% -------------------------------------------------------------------------
% SURVEY
% -------------------------------------------------------------------------
S.session_date = NaT;
for j = 1:length(raweeg_files)
    raw_pid = extractBefore(raweeg_files(j).name, '.eeg');  % adjust if needed
    if strcmp(extractBefore(S.rootID, '_ds_500hz'), raw_pid)
        S.session_date = dateshift(raweeg_files(j).correct_date, 'start', 'day');
        break
    end
end

if isnat(S.session_date)
    error('No session date found for %s', session_ID);
end

sur_row_idx = find(dateshift(S.session_date, 'start', 'day') == ...
                   dateshift(SUR_WIDE.SESH_DATE, 'start', 'day'));

if isempty(sur_row_idx)
    error('No survey row found for %s', session_ID);
end
if ~strcmp(char(SUR_WIDE.PID(sur_row_idx)), S.PID)
    warning('Survey PID mismatch for %s', S.PID);
end
if ~(S.sesh_type == SUR_WIDE.SESH_TYPE(sur_row_idx))
    warning('Session type mismatch for %s', S.PID);
end

S.sur_wide    = SUR_WIDE(sur_row_idx, :);
S.asc_ratings = S.sur_wide{1, 7:16};
S.sur_added   = true;

% -------------------------------------------------------------------------
% SMT (optional)
% -------------------------------------------------------------------------
if have_smt
    smt_row_idx = find(dateshift(S.session_date, 'start', 'day') == ...
                       dateshift(SMT_TBL.date, 'start', 'day'));

    if ~isempty(smt_row_idx)
        if ~strcmp(char(SMT_TBL.PID(smt_row_idx)), S.PID)
            warning('SMT PID mismatch for %s', S.PID);
        end
        S.SMT       = SMT_TBL(smt_row_idx, :);
        S.smt_added = true;
    else
        warning('No SMT found for %s -- continuing without it', session_ID);
        S.smt_added = false;
    end
else
    S.smt_added = false;
end

% -------------------------------------------------------------------------
% SAVE
% -------------------------------------------------------------------------
save(save_path, 'S', '-v7.3');
fprintf('Saved: %s\n', save_path);

end