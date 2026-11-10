function RUN_FREQ_ANALYSIS_fun(merged_fname, FREQ_OUT, SCRIPTS_DIR)

% 8 Aug update: reverted median -> mean (delta/gamma discrepancy found
% with median); IRASA removed entirely (no longer used)

% 3 Sept update: get broadband periodic - aperiodic 

addpath 'PATH_TO_FIELDTRIP';
ft_defaults;

addpath(genpath(SCRIPTS_DIR));

% --- skip check
[froot, fname] = fileparts(merged_fname);
save_path  = fullfile(FREQ_OUT, [fname '_freq.mat']);

if isfile(save_path)
    fprintf('Already processed: %s\n', fname);
    return
end

fprintf('\nProcessing: %s\n', fname);

% --- load
tmp = load(merged_fname, 'S');
SP  = tmp.S;

% --- params
bands(1).name = 'delta'; bands(1).range = [1 4];
bands(2).name = 'theta'; bands(2).range = [4 8];
bands(3).name = 'alpha'; bands(3).range = [8 13];
bands(4).name = 'beta'; bands(4).range = [13 30];
bands(5).name = 'gamma'; bands(5).range = [30 45];

nCond = 7;
nBand = numel(bands);

epochparams = struct();
epochparams.length  = 5;
epochparams.overlap = 0;

% --- <<<<< matlab random number generator 
rng(42)

% Loosened to catch only drastic/unambiguous artifacts -- starting points,
% recalibrate once you've checked which specific criterion was responsible
% for rejecting the good-fit epochs you noticed.
rejectparams = struct();
rejectparams.abs_thresh = 150;
rejectparams.p2p_thresh = 180;
rejectparams.epoch_std  = 35;
rejectparams.var_ratio  = 3;

% --- frequency configs
cfg_freq            = [];
cfg_freq.method     = 'mtmfft';
cfg_freq.output     = 'pow';
cfg_freq.taper      = 'dpss';
cfg_freq.tapsmofrq  = 0.5;   % 1 Hz smoothing
cfg_freq.foilim     = [1 45];
cfg_freq.pad        = 'nextpow2';
cfg_freq.keeptrials = 'yes';   % per-epoch power

cfg_freq_fooof                         = cfg_freq;
cfg_freq_fooof.output                  = 'fooof_aperiodic';
cfg_freq_fooof.keeptrials              = 'no';   % FOOOF needs an averaged spectrum
cfg_freq_fooof.fooof                   = [];
cfg_freq_fooof.fooof.freq_range        = [1 45];
cfg_freq_fooof.fooof.peak_width_limits = [1 12];
cfg_freq_fooof.fooof.peak_threshold    = 1.5;
cfg_freq_fooof.fooof.aperiodic_mode    = 'fixed';

% --- init R struct: copy metadata
out_fields  = {'data_pass','data_act','sur_acute','aux_added'};
all_fields  = fieldnames(SP);
keep_fields = all_fields(~ismember(all_fields, out_fields));

R = struct();
for f = 1:numel(keep_fields)
    R.(keep_fields{f}) = SP.(keep_fields{f});
end

R.file_root = froot;
R.freqparams.rejectparams = rejectparams;
R.freqparams.epochparams = epochparams;
R.freqparams.cfg_freq_fooof = cfg_freq_fooof;

if isfield(SP.data_pass,'trial_labels')
    data_pass = rmfield(SP.data_pass,'trial_labels');
else
    data_pass = SP.data_pass;
end
clear SP tmp;

% --- main loop
for c = 1:nCond

    fprintf('  Condition %d/%d\n', c, nCond);

    cfg         = [];
    cfg.trials  = c;
    cfg.channel = data_pass.label(1:63);
    data_seg    = ft_selectdata(cfg, data_pass);

    % ============================================================
    %  EPOCH SELECTION (70% floor / 80% cap, random subsample)
    % ============================================================
    [data_ep_cln, ep_info] = fun_epoch_reject(data_seg, epochparams, rejectparams);

    trial_dur_sec = numel(data_seg.time{1}) / data_seg.fsample;
    n_max_epochs  = floor(trial_dur_sec / epochparams.length);
    min_n_epochs  = floor(0.70 * n_max_epochs);
    max_n_epochs  = floor(0.80 * n_max_epochs);
    n_clean       = numel(data_ep_cln.trial);

    R.n_epochs_min_required(c) = min_n_epochs;
    R.n_epochs_max_allowed(c)  = max_n_epochs;
    R.n_epochs_clean(c)        = n_clean;

    if n_clean < min_n_epochs
        error('not enough epochs for this dataset')
    end

    target_n_epochs = min(n_clean, max_n_epochs);
    R.n_epochs_target(c) = target_n_epochs;

    sel_idx        = sort(randperm(n_clean, target_n_epochs));
    cfg_sel        = [];
    cfg_sel.trials = sel_idx;
    data_ep_cln    = ft_selectdata(cfg_sel, data_ep_cln);

    R.n_epochs_used(c) = numel(data_ep_cln.trial);

    % ============================================================
    %  PER-EPOCH POWER (source data for the epoch-averaged convenience
    %  fields added at the end of this section)
    % ============================================================
    freq_ep   = ft_freqanalysis(cfg_freq, data_ep_cln);
    freq_vec  = freq_ep.freq;
    nFreq     = numel(freq_vec);
    nEpochs   = numel(data_ep_cln.trial);
    nchan     = numel(freq_ep.label);

    pwr_3d = freq_ep.powspctrm;   % nEpochs x nChan x nFreq

    for b = 1:nBand
        foi    = freq_vec >= bands(b).range(1) & freq_vec <= bands(b).range(2);
        f_band = freq_vec(foi);

        % ep_band_raw: mean across freq bins, per epoch -> nEpochs x nChan
        R.ep_band_raw.(bands(b).name){c} = squeeze(mean(pwr_3d(:,:,foi), 3));

        % ep_band_total: trapz (area under curve), per epoch -> nEpochs x nChan
        ep_trapz = zeros(nEpochs, nchan);
        for ep = 1:nEpochs
            for ch = 1:nchan
                ep_trapz(ep,ch) = trapz(f_band, squeeze(pwr_3d(ep,ch,foi)));
            end
        end
        R.ep_band_total.(bands(b).name){c} = ep_trapz;

        % across-epoch averages, for convenient direct access. NOTE: NOT
        % the same quantity as band_total_auc below -- that's
        % trapz(mean-across-epochs-spectrum), these are mean(trapz-per-
        % epoch). Order of averaging vs integration differs (Jensen's
        % inequality), close but not numerically identical.
        R.band_raw_epochmean.(bands(b).name){c}   = mean(R.ep_band_raw.(bands(b).name){c}, 1, 'omitnan')';
        R.band_total_epochmean.(bands(b).name){c} = mean(ep_trapz, 1, 'omitnan')';
    end

    % ============================================================
    %  SESSION-AVERAGED SPECTRUM + FOOOF FIT
    % ============================================================

    % >>> MEASURE 1: RAW PSD (full spectrum, session-averaged across epochs) <<<
    avg_pwr = squeeze(mean(pwr_3d, 1));   % nChan x nFreq
    R.broad_pwr{c} = avg_pwr;             % <-- MEASURE 1: raw PSD

    % >>> MEASURE 2: raw PSD, area under curve (per band) <<<
    for b = 1:nBand
        foi    = freq_vec >= bands(b).range(1) & freq_vec <= bands(b).range(2);
        f_band = freq_vec(foi);
        R.band_total_auc.(bands(b).name){c} = trapz(f_band, avg_pwr(:,foi), 2);   % <-- MEASURE 2
    end

    R.freq_vec = freq_vec;

    subj_fractal = ft_freqanalysis(cfg_freq_fooof, data_ep_cln);

    if ~isfield(subj_fractal.fooofparams, 'r_squared') || ~isfield(subj_fractal.fooofparams, 'error')
        error('subj_fractal.fooofparams does not have r_squared/error fields -- run fieldnames() on it and update this script.');
    end

    % >>> MEASURE 3 (values): exponent + offset, fitted to the raw PSD <<<
    exp_map    = zeros(nchan, 1);
    offset_map = zeros(nchan, 1);
    R.fooof_r2{c}    = zeros(nchan, 1);   % fit quality -- KEEP, used extensively for QC/confound checks
    R.fooof_error{c} = zeros(nchan, 1);   % fit error -- KEEP, paired QC metric alongside r2

    for ch = 1:nchan
        p = subj_fractal.fooofparams(ch).aperiodic_params;
        offset_map(ch) = p(1);
        exp_map(ch)    = p(end);
        R.fooof_r2{c}(ch)    = subj_fractal.fooofparams(ch).r_squared;
        R.fooof_error{c}(ch) = subj_fractal.fooofparams(ch).error;
    end
    R.exponent{c} = exp_map;    % <-- MEASURE 3 (values): exponent
    R.offset{c}   = offset_map; % <-- MEASURE 3 (values): offset

    % >>> MEASURE 3 (line to plot): the fitted aperiodic curve <<<
    % subj_fractal.powspctrm IS ALREADY the fitted aperiodic curve in
    % LINEAR power -- mathematically identical to manually reconstructing
    % 10.^(offset - exponent*log10(freq)) for aperiodic_mode='fixed'.
    R.aper_spectrum{c}  = log10(subj_fractal.powspctrm);   % log10; 10.^this = linear power
    R.aper_curve_lin{c} = subj_fractal.powspctrm;           % linear power (uV^2) directly

    % >>> MEASURE 4: area under the exponent/aperiodic curve (per band) <<<
    for b = 1:nBand
        foi    = freq_vec >= bands(b).range(1) & freq_vec <= bands(b).range(2);
        f_band = freq_vec(foi);
        R.band_aper_auc.(bands(b).name){c} = trapz(f_band, subj_fractal.powspctrm(:,foi), 2);   % <-- MEASURE 4
    
        
    end

    % >>> MEASURE 5: residual / "foofed" PSD (full spectrum) <<<
    osc_spec = log10(avg_pwr) - log10(subj_fractal.powspctrm);
    R.osc_pwr{c} = osc_spec;   % <-- MEASURE 5: foofed PSD (log-ratio, full spectrum)

    % >>> MEASURE 6: "area under" the foofed PSD (per band) <<<
    for b = 1:nBand
        foi = freq_vec >= bands(b).range(1) & freq_vec <= bands(b).range(2);
        R.band_fooof.(bands(b).name){c} = mean(osc_spec(:, foi), 2);   % mean, NOT trapz -- see band_fooof_auc for the trapz version
        f_band = freq_vec(foi);
        R.band_fooof_auc.(bands(b).name){c} = trapz(f_band, osc_spec(:,foi), 2);   % <-- MEASURE 6 (trapz, consistent with measures 2/4)
    end

    % >>> MEASURE 7: BROADBAND (whole 1-45Hz range, not per-band) area under
    % curve -- total, aperiodic, and foofed/periodic. Same trapz logic as
    % Measures 2/4/6 above, just integrated over the FULL freq_vec range
    % instead of a band-restricted foi. Reuses avg_pwr (Measure 1),
    % subj_fractal.powspctrm (Measure 3's aperiodic curve), and osc_spec
    % (Measure 5) -- all already computed above, nothing new to fit. <<<
    R.broad_total_auc{c} = trapz(freq_vec, avg_pwr, 2);                    % <-- MEASURE 7a: whole-spectrum raw PSD AUC
    R.broad_aper_auc{c}  = trapz(freq_vec, subj_fractal.powspctrm, 2);     % <-- MEASURE 7b: whole-spectrum aperiodic AUC
    R.broad_fooof_auc{c} = trapz(freq_vec, osc_spec, 2);                   % <-- MEASURE 7c: whole-spectrum foofed/periodic AUC
end

% -------------------------------------------------------------------------
% PHYSIO FREQ
% -------------------------------------------------------------------------
physio = get_aux_freq(data_pass, nCond);
R.heart_rate    = physio.heart_rate;
R.hrv_sdnn      = physio.hrv_sdnn;
R.hrv_rmssd     = physio.hrv_rmssd;
R.hrv_pnn50     = physio.hrv_pnn50;
R.edr_amp       = physio.edr_amp;
R.edr_resp_rate = physio.edr_resp_rate;
R.edr_resp_std  = physio.edr_resp_std;

% -------------------------------------------------------------------------
% SAVE
% -------------------------------------------------------------------------
fprintf('Saving: %s\n', save_path);
save(save_path, 'R', '-v7.3');
fprintf('Done: %s\n', fname);

end