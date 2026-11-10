function r_peaks = detect_r_peaks(ecg_signal, fs, cfg)
% DETECT_R_PEAKS  Find R-peak timestamps in an ECG channel.
%
%   r_peaks = detect_r_peaks(ecg_signal, fs, cfg)
%
% Simple Pan-Tompkins-style pipeline: bandpass -> rectify/square ->
% moving-average envelope -> robust z-score peak picking. The refractory
% period is intentionally short by resting-EEG standards because
% breathwork can drive heart rate well above typical resting range (fast
% paced breathing / tachycardia in your R2/R3 segments) -- a refractory
% period tuned for ~150 bpm will silently miss beats exactly in the
% segments you care about most.
%
% cfg fields (all optional):
%   bp_band       - bandpass range emphasizing QRS complex, Hz (default [5 15])
%   smooth_ms     - moving-average window after rectification, ms (default 150)
%   z_thresh      - robust z-score threshold for peak detection (default 4)
%   min_isi_sec   - minimum inter-beat interval, seconds (default 0.28,
%                   i.e. caps detectable rate at ~214 bpm -- adjust down
%                   further only if you have reason to expect faster)

if nargin < 3, cfg = struct(); end
if ~isfield(cfg,'bp_band'),     cfg.bp_band     = [5 15]; end
if ~isfield(cfg,'smooth_ms'),   cfg.smooth_ms   = 150;    end
if ~isfield(cfg,'z_thresh'),    cfg.z_thresh    = 4;      end
if ~isfield(cfg,'min_isi_sec'), cfg.min_isi_sec = 0.28;   end

sig = double(ecg_signal);
sig = sig - mean(sig);

% --- Bandpass to emphasize QRS complex
nyq = fs/2;
[b,a] = butter(3, cfg.bp_band/nyq, 'bandpass');
sig_bp = filtfilt(b, a, sig);

% --- Rectify (square) and smooth into an envelope
sig_sq = sig_bp .^ 2;
smooth_n = max(1, round(cfg.smooth_ms/1000 * fs));
envelope = movmean(sig_sq, smooth_n);

% --- Robust z-score peak picking
med  = median(envelope);
madv = mad(envelope, 1) / 0.6745;
if madv == 0
    r_peaks = [];
    warning('detect_r_peaks:flatSignal', 'ECG envelope appears flat; no R-peaks detected.');
    return
end
z = (envelope - med) / madv;

min_dist = round(cfg.min_isi_sec * fs);
[~, locs] = findpeaks(z, 'MinPeakHeight', cfg.z_thresh, 'MinPeakDistance', min_dist);

r_peaks = locs(:);

if isempty(r_peaks)
    warning('detect_r_peaks:noPeaks', 'No R-peaks detected above z_thresh=%.1f; check ECG channel quality or lower threshold.', cfg.z_thresh);
else
    % Sanity-check implied heart rate -- flag if it looks physiologically implausible,
    % which usually means the threshold/band needs adjusting rather than a real HR.
    ibi_sec = diff(r_peaks) / fs;
    implied_bpm = 60 ./ median(ibi_sec);
    if implied_bpm < 35 || implied_bpm > 220
        warning('detect_r_peaks:implausibleRate', ...
            'Implied heart rate %.0f bpm looks physiologically implausible; check ECG signal quality.', implied_bpm);
    end
end
end