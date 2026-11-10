function blink_events = detect_blink_events(eog_signal, fs, cfg)
% DETECT_BLINK_EVENTS  Find discrete blink timestamps in an EOG-cheek channel.
%
%   blink_events = detect_blink_events(eog_signal, fs, cfg)
%
% Uses a robust (MAD-based) z-score threshold on the rectified, smoothed
% signal, with a refractory period so double-detections on one blink's
% rise/fall aren't counted twice.
%
% cfg fields (all optional):
%   z_thresh        - robust z-score threshold for peak detection (default 4)
%   min_isi_sec     - minimum inter-blink interval, seconds (default 0.4)
%   smooth_ms       - smoothing window before peak detection, ms (default 50)

if nargin < 3, cfg = struct(); end
if ~isfield(cfg,'z_thresh'),    cfg.z_thresh    = 4;   end
if ~isfield(cfg,'min_isi_sec'), cfg.min_isi_sec = 0.4; end
if ~isfield(cfg,'smooth_ms'),   cfg.smooth_ms   = 50;  end

sig = double(eog_signal);
sig = sig - median(sig);

smooth_n = max(1, round(cfg.smooth_ms/1000 * fs));
sig_smooth = movmean(abs(sig), smooth_n);

med  = median(sig_smooth);
madv = mad(sig_smooth, 1) / 0.6745;
if madv == 0
    blink_events = [];
    warning('detect_blink_events:flatSignal', 'EOG signal appears flat; no blinks detected.');
    return
end
z = (sig_smooth - med) / madv;

min_dist = round(cfg.min_isi_sec * fs);
[~, locs] = findpeaks(z, 'MinPeakHeight', cfg.z_thresh, 'MinPeakDistance', min_dist);

blink_events = locs(:);

if isempty(blink_events)
    warning('detect_blink_events:noBlinks', 'No blink events detected above z_thresh=%.1f; check EOG channel quality or lower threshold.', cfg.z_thresh);
end
end