
function [icTypes, diag] = get_component_types(EEG, AUX, cfg)
% GET_COMPONENT_TYPES  Assign one type per independent component.
%
%   [icTypes, diag] = get_component_types(EEG, AUX, cfg)
%
% INPUTS
%   EEG  - EEGLAB struct with ICA already run and pop_iclabel already applied
%          (EEG.etc.ic_classification.ICLabel must exist)
%   AUX  - EEGLAB struct with the true EOG-cheek channel (and ECG if you
%          want it for diagnostics/plots; ECG is not used for typing here)
%   cfg  - struct of thresholds / options, see DEFAULTS below. Pass [] or
%          partial struct to use defaults for missing fields.
%
% OUTPUT
%   icTypes - n_comps x 1 cell array of strings:
%             'brain' | 'eye' | 'cardiac' | 'muscle' | 'ambiguous'
%   diag    - struct with every intermediate metric, for review/plotting/
%             calibration and for saving into data_status
%
% TYPE LOGIC (evaluated in this precedence order per component):
%   1. eye:    ICLabel eye-prob > cfg.eye_iclabel_thresh (strong call, trusted outright)
%              OR at least cfg.eye_min_votes of the following 4 pass:
%                - ICLabel eye-prob > cfg.eye_iclabel_corrob_thresh (mild lean)
%                - blink-locked score > cfg.blink_score_thresh
%                - kurtosis in blink window > cfg.blink_kurtosis_thresh
%                - frontal topo ratio > cfg.frontal_topo_ratio_thresh
%              This is a vote count, not a strict AND of all blink-morphology
%              criteria -- a component with strong evidence on 3 of 4 fronts
%              is typed eye even if one metric (e.g. blink-locked score) is
%              borderline, since detection-threshold changes elsewhere
%              (e.g. loosening blink-event detection sensitivity) can dilute
%              that one metric without changing the others.
%              If this evidence conflicts with a >90%-confidence ICLabel
%              brain call, the component is typed 'ambiguous' for review
%              rather than deleted (see high_confidence_brain guard below).
%              Whole-window EOG correlation (eog_corr) is kept only as a
%              diagnostic, not a decision criterion.
%   2. cardiac: ICLabel heart-prob > cfg.heart_iclabel_thresh
%              OR R-peak-locked evidence: amplitude specifically elevated
%              at detected ECG R-peaks (cardiac_score) AND consistently so
%              across beats, i.e. low coefficient of variation (cardiac_cv)
%              -- both required, since score alone can be inflated by a
%              few coincidentally large epochs. Same conflict guard as eye:
%              if this evidence conflicts with a >90%-confidence ICLabel
%              brain call, typed 'ambiguous' rather than deleted.
%   3. muscle: ICLabel muscle-prob > cfg.muscle_iclabel_thresh
%              AND (spectral slope in muscle band > cfg.muscle_slope_thresh
%                   OR temporal/central topo ratio > cfg.muscle_topo_ratio_thresh)
%              -> if ICLabel says muscle but neither corroborator agrees,
%                 falls through to ambiguous (denoised, not deleted)
%   4. brain:  ICLabel brain-prob > cfg.brain_iclabel_thresh
%              (only checked if not already eye/cardiac/muscle)
%   5. channel/line noise: ICLabel noise-prob > cfg.noise_iclabel_thresh
%              -> treated as hard-removal, folded into 'muscle' bucket in
%                 the type output for simplicity (both are non-neural,
%                 both get fully removed). See diag.is_noise if you want
%                 to keep them visually distinct.
%   6. everything else -> ambiguous

if nargin < 3 || isempty(cfg), cfg = struct(); end
cfg = set_default(cfg, 'eye_iclabel_thresh',      0.70);
cfg = set_default(cfg, 'eye_iclabel_corrob_thresh', 0.40);  % mild ICLabel lean counts as one vote, doesn't need to clear the strong-call threshold alone
cfg = set_default(cfg, 'eye_min_votes',           3);       % how many of {iclabel-lean, blink_score, kurtosis, frontal} must pass
cfg = set_default(cfg, 'eog_corr_thresh',         0.30);   % now a secondary/diagnostic metric, see blink-locked score below
cfg = set_default(cfg, 'blink_window_sec',        300);    % first N seconds used for blink detection (your eyes-open period)
cfg = set_default(cfg, 'blink_score_thresh',      3.0);    % blink-locked amplitude ratio vs null; >1 = elevated at blinks
cfg = set_default(cfg, 'blink_kurtosis_thresh',   5.0);    % excess kurtosis in blink window; blinks are sparse+large -> high kurtosis
cfg = set_default(cfg, 'frontal_topo_ratio_thresh', 1.5);  % frontal/rest |weight| ratio
cfg = set_default(cfg, 'frontal_chans', {'Fp1','Fp2','AF7','AF8','Fpz','AF3','AF4'});
cfg = set_default(cfg, 'brain_override_prob',     0.90);   % below this, blink evidence can override ICLabel's brain call; above this, ICLabel wins unless blink evidence is very strong
cfg = set_default(cfg, 'heart_iclabel_thresh',    0.70);
cfg = set_default(cfg, 'cardiac_score_thresh',    3.0);     % R-peak-locked amplitude ratio vs null
cfg = set_default(cfg, 'cardiac_cv_thresh',       0.6);     % coefficient of variation ceiling; below this = consistently locked to beats
cfg = set_default(cfg, 'cardiac_window_sec',      300);     % window used for R-peak detection + scoring; consider using a window that spans a segment with stable-ish heart rate for detection purposes, scoring itself tolerates HR changes fine
cfg = set_default(cfg, 'muscle_iclabel_thresh',   0.75);
cfg = set_default(cfg, 'muscle_slope_thresh',    -1.00);  % slope in 20-45Hz log-log fit; less negative = flatter = more muscle-like
cfg = set_default(cfg, 'muscle_topo_ratio_thresh', 1.50); % peripheral/central |weight| ratio
cfg = set_default(cfg, 'brain_iclabel_thresh',    0.70);
cfg = set_default(cfg, 'noise_iclabel_thresh',    0.75);
cfg = set_default(cfg, 'muscle_band',             [20 45]);
cfg = set_default(cfg, 'corr_window_sec',         180);    % seconds used for AUX correlation
cfg = set_default(cfg, 'peripheral_chans', {'T7','T8','FT7','FT8','TP7','TP8','F7','F8','FT9','FT10','TP9','TP10'});
cfg = set_default(cfg, 'central_chans',    {'Cz','C1','C2','C3','C4','FC1','FC2','CP1','CP2','FCz','CPz'});

classes         = EEG.etc.ic_classification.ICLabel.classes;
classifications = EEG.etc.ic_classification.ICLabel.classifications;
n_comps         = size(classifications, 1);

ica_act = eeg_getdatact(EEG, 'component', 1:n_comps);
fs      = EEG.srate;

col = @(name) classifications(:, strcmpi(classes, name));
eye_prob    = col('Eye');
muscle_prob = col('Muscle');
brain_prob  = col('Brain');
if any(strcmpi(classes,'Heart'))
    heart_prob = col('Heart');
else
    heart_prob = zeros(n_comps,1);
    warning('get_component_types:noHeartClass', 'ICLabel classes do not include "Heart"; relying on R-peak-locked evidence alone for cardiac typing.');
end
if any(strcmpi(classes,'Channel Noise'))
    noise_prob = col('Channel Noise');
else
    noise_prob = zeros(n_comps,1);
end

% ---------------- EOG-cheek: blink event detection + blink-locked scoring ----------------
eog_ch_idx = find(strcmpi({AUX.chanlocs.labels}, 'EOG') | strcmpi({AUX.chanlocs.labels}, 'EMG'));


n_corr = min(round(cfg.corr_window_sec * fs), min(size(ica_act,2), AUX.pnts));
n_blink_win = min(round(cfg.blink_window_sec * fs), min(size(ica_act,2), AUX.pnts));

eog_corr          = zeros(n_comps,1);   % kept as secondary diagnostic
blink_score       = nan(n_comps,1);
blink_n_events    = 0;
ic_kurtosis_blink  = nan(n_comps,1);

if ~isempty(eog_ch_idx)
    eog_sig_corrwin = double(AUX.data(eog_ch_idx(1), 1:n_corr));
    for ic = 1:n_comps
        r = corrcoef(double(ica_act(ic,1:n_corr)), eog_sig_corrwin);
        eog_corr(ic) = abs(r(1,2));
    end

    eog_sig_blinkwin = double(AUX.data(eog_ch_idx(1), 1:n_blink_win));
    blink_events = detect_blink_events(eog_sig_blinkwin, fs, struct());
    blink_n_events = numel(blink_events);

    if blink_n_events >= 5   % need a reasonable number of events for a stable estimate
        for ic = 1:n_comps
            ic_win = double(ica_act(ic, 1:n_blink_win));
            blink_score(ic)      = blink_locked_score(ic_win, blink_events, fs, struct());
            ic_kurtosis_blink(ic)= kurtosis(ic_win) - 3;  % excess kurtosis
        end
    else
        warning('get_component_types:fewBlinks', ...
            'Only %d blink events detected in first %.0fs; blink-locked scoring unreliable, falling back to ICLabel + whole-window corr only.', ...
            blink_n_events, cfg.blink_window_sec);
    end
else
    warning('get_component_types:noEOG', 'No EOG channel found in AUX set; eye typing relies on ICLabel only.');
end

% ---------------- ECG: R-peak detection + cardiac-locked scoring ----------------
ecg_ch_idx = find(strcmpi({AUX.chanlocs.labels}, 'ECG'), 1);
if isempty(ecg_ch_idx)
    ecg_ch_idx = find(strcmpi({AUX.chanlocs.labels}, 'ECG2'), 1);  % fall back to second ECG channel if primary absent
end

n_cardiac_win = min(round(cfg.cardiac_window_sec * fs), min(size(ica_act,2), AUX.pnts));

cardiac_score    = nan(n_comps,1);
cardiac_cv       = nan(n_comps,1);
r_peak_n_events  = 0;

if ~isempty(ecg_ch_idx)
    ecg_sig_win = double(AUX.data(ecg_ch_idx, 1:n_cardiac_win));
    r_peaks = detect_r_peaks(ecg_sig_win, fs, struct());
    r_peak_n_events = numel(r_peaks);

    if r_peak_n_events >= 10   % want a reasonably stable CV estimate, cardiac events are frequent so this is a low bar
        for ic = 1:n_comps
            ic_win = double(ica_act(ic, 1:n_cardiac_win));
            [cardiac_score(ic), cardiac_cv(ic)] = cardiac_locked_score(ic_win, r_peaks, fs, struct());
        end
    else
        warning('get_component_types:fewRPeaks', ...
            'Only %d R-peaks detected in first %.0fs; cardiac-locked scoring unreliable, falling back to ICLabel Heart class only.', ...
            r_peak_n_events, cfg.cardiac_window_sec);
    end
else
    warning('get_component_types:noECG', 'No ECG channel found in AUX set; cardiac typing relies on ICLabel Heart class only.');
end

% ---------------- Spectral slope (muscle band) ----------------
muscle_slope = nan(n_comps,1);
nfft   = min(2^nextpow2(fs*2), size(ica_act,2));
for ic = 1:n_comps
    [pxx, f] = pwelch(double(ica_act(ic,:)), hamming(nfft), round(nfft/2), nfft, fs);
    band_idx = f >= cfg.muscle_band(1) & f <= cfg.muscle_band(2);
    logf = log10(f(band_idx));
    logp = log10(pxx(band_idx));
    p = polyfit(logf, logp, 1);
    muscle_slope(ic) = p(1);  % closer to 0 / positive = flatter = more muscle-like
end

% ---------------- Topography ratios (peripheral/central for muscle, frontal for eye) ----------------
all_labels = upper({EEG.chanlocs.labels});
peri_idx  = find(ismember(all_labels, upper(cfg.peripheral_chans)));
cent_idx  = find(ismember(all_labels, upper(cfg.central_chans)));
front_idx = find(ismember(all_labels, upper(cfg.frontal_chans)));
rest_idx  = setdiff(1:numel(all_labels), front_idx);

topo_ratio = nan(n_comps,1);
if ~isempty(peri_idx) && ~isempty(cent_idx)
    topo = abs(EEG.icawinv);  % channels x comps
    peri_mean = mean(topo(peri_idx,:), 1);
    cent_mean = mean(topo(cent_idx,:), 1);
    topo_ratio = (peri_mean ./ max(cent_mean, eps))';
else
    warning('get_component_types:noChanGroups', 'Peripheral/central channel groups not found in chanlocs; muscle topo ratio unavailable.');
end

frontal_topo_ratio = nan(n_comps,1);
if ~isempty(front_idx) && ~isempty(rest_idx)
    topo = abs(EEG.icawinv);
    front_mean = mean(topo(front_idx,:), 1);
    rest_mean  = mean(topo(rest_idx,:), 1);
    frontal_topo_ratio = (front_mean ./ max(rest_mean, eps))';
else
    warning('get_component_types:noFrontalChans', 'Frontal channel group not found in chanlocs; eye topo ratio unavailable.');
end

% ---------------- Assemble types ----------------
icTypes = cell(n_comps,1);
is_eye     = false(n_comps,1);
is_cardiac = false(n_comps,1);
is_muscle  = false(n_comps,1);
is_brain   = false(n_comps,1);
is_noise   = false(n_comps,1);
is_ambig   = false(n_comps,1);
muscle_corroborated = false(n_comps,1);
eye_votes_all = nan(n_comps,1);

for ic = 1:n_comps
    % --- Eye decision ---
    % Strong ICLabel eye call: trust it outright.
    iclabel_says_eye = eye_prob(ic) > cfg.eye_iclabel_thresh;

    % Otherwise, require blink-morphology evidence: the IC must (a) spike
    % specifically at detected blink timestamps (not just correlate with
    % a shared slow rhythm like alpha), (b) look sparse/peaky (kurtosis),
    % and (c) load frontally. Any one of these alone is not enough.
    has_blink_score = ~isnan(blink_score(ic));
    eye_votes = 0;
    if eye_prob(ic) > cfg.eye_iclabel_corrob_thresh, eye_votes = eye_votes + 1; end
    if has_blink_score && blink_score(ic)       > cfg.blink_score_thresh,      eye_votes = eye_votes + 1; end
    if has_blink_score && ic_kurtosis_blink(ic) > cfg.blink_kurtosis_thresh,   eye_votes = eye_votes + 1; end
    if ~isnan(frontal_topo_ratio(ic)) && frontal_topo_ratio(ic) > cfg.frontal_topo_ratio_thresh, eye_votes = eye_votes + 1; end
    blink_evidence = has_blink_score && eye_votes >= cfg.eye_min_votes;
    eye_votes_all(ic) = eye_votes;

    % Guard: if ICLabel is already very confident this is brain, only let
    % blink evidence override it when that evidence is unambiguous (all
    % three criteria above), never on whole-window correlation alone.
    high_confidence_brain = brain_prob(ic) > cfg.brain_override_prob;

    if iclabel_says_eye || (blink_evidence && ~high_confidence_brain)
        icTypes{ic} = 'eye';
        is_eye(ic) = true;
        continue
    elseif blink_evidence && high_confidence_brain
        % Conflicting strong evidence both ways -- don't silently pick one.
        % Flag as ambiguous so it gets human review / gentle denoising
        % rather than full deletion of a component ICLabel is very sure
        % is neural.
        icTypes{ic} = 'ambiguous';
        is_ambig(ic) = true;
        continue
    end

    % --- Cardiac decision ---
    iclabel_says_heart = heart_prob(ic) > cfg.heart_iclabel_thresh;

    has_cardiac_score = ~isnan(cardiac_score(ic));
    cardiac_evidence = has_cardiac_score && ...
        cardiac_score(ic) > cfg.cardiac_score_thresh && ...
        cardiac_cv(ic)    < cfg.cardiac_cv_thresh;   % low CV = consistently locked to (almost) every beat

    if iclabel_says_heart || (cardiac_evidence && ~high_confidence_brain)
        icTypes{ic} = 'cardiac';
        is_cardiac(ic) = true;
        continue
    elseif cardiac_evidence && high_confidence_brain
        icTypes{ic} = 'ambiguous';
        is_ambig(ic) = true;
        continue
    end

    if muscle_prob(ic) > cfg.muscle_iclabel_thresh
        slope_flags_muscle = ~isnan(muscle_slope(ic)) && muscle_slope(ic) > cfg.muscle_slope_thresh;
        topo_flags_muscle  = ~isnan(topo_ratio(ic))   && topo_ratio(ic)   > cfg.muscle_topo_ratio_thresh;
        if slope_flags_muscle || topo_flags_muscle
            icTypes{ic} = 'muscle';
            is_muscle(ic) = true;
            muscle_corroborated(ic) = true;
            continue
        end
        % ICLabel says muscle but no corroboration -> fall through to ambiguous
    end

    if noise_prob(ic) > cfg.noise_iclabel_thresh
        icTypes{ic} = 'muscle';  % lumped into hard-removal bucket; see diag.is_noise
        is_noise(ic) = true;
        continue
    end

    if brain_prob(ic) > cfg.brain_iclabel_thresh
        icTypes{ic} = 'brain';
        is_brain(ic) = true;
        continue
    end

    icTypes{ic} = 'ambiguous';
    is_ambig(ic) = true;
end

diag = struct();
diag.cfg                  = cfg;
diag.classes               = classes;
diag.classifications       = classifications;
diag.eye_prob              = eye_prob;
diag.muscle_prob           = muscle_prob;
diag.brain_prob            = brain_prob;
diag.noise_prob            = noise_prob;
diag.eog_corr              = eog_corr;
diag.blink_score           = blink_score;
diag.blink_n_events        = blink_n_events;
diag.ic_kurtosis_blink     = ic_kurtosis_blink;
diag.frontal_topo_ratio    = frontal_topo_ratio;
diag.eye_votes             = eye_votes_all;
diag.heart_prob            = heart_prob;
diag.cardiac_score         = cardiac_score;
diag.cardiac_cv            = cardiac_cv;
diag.r_peak_n_events       = r_peak_n_events;
diag.is_cardiac            = is_cardiac;
diag.muscle_slope          = muscle_slope;
diag.topo_ratio            = topo_ratio;
diag.is_eye                = is_eye;
diag.is_muscle             = is_muscle;
diag.is_brain              = is_brain;
diag.is_noise              = is_noise;
diag.is_ambiguous          = is_ambig;
diag.muscle_corroborated   = muscle_corroborated;
diag.icTypes               = icTypes;

fprintf('\n--- Component typing summary (%s) ---\n', EEG.subject);
fprintf('  blink events detected in first %.0fs: %d\n', cfg.blink_window_sec, blink_n_events);
fprintf('  R-peaks detected in first %.0fs:      %d\n', cfg.cardiac_window_sec, r_peak_n_events);
fprintf('  eye:       %d\n', sum(is_eye));
fprintf('  cardiac:   %d\n', sum(is_cardiac));
fprintf('  muscle:    %d  (%d ICLabel+corroborated, %d channel/line noise)\n', ...
    sum(is_muscle)+sum(is_noise), sum(muscle_corroborated), sum(is_noise));
fprintf('  brain:     %d\n', sum(is_brain));
fprintf('  ambiguous: %d  (includes %d muscle-flagged-by-ICLabel-but-uncorroborated)\n', ...
    sum(is_ambig), sum(muscle_prob > cfg.muscle_iclabel_thresh & is_ambig));

end

% -------------------------------------------------------------------
function cfg = set_default(cfg, field, value)
if ~isfield(cfg, field) || isempty(cfg.(field))
    cfg.(field) = value;
end
end

