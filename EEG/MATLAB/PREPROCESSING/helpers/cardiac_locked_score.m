function [score, regularity, diag] = cardiac_locked_score(ic_signal, r_peaks, fs, cfg)
% CARDIAC_LOCKED_SCORE  Quantify whether an IC's amplitude is specifically
% and consistently elevated at R-peak timestamps.
%
%   [score, regularity, diag] = cardiac_locked_score(ic_signal, r_peaks, fs, cfg)
%
% Returns two complementary metrics:
%   score      - (median abs amplitude in R-peak-locked epochs) /
%                (median abs amplitude in null distribution of shuffled
%                epochs). Same logic as blink_locked_score: values well
%                above 1 mean the component reacts specifically at heart
%                beats, not just amplitude that happens to be high overall.
%   regularity - coefficient of variation (CV = std/mean) of the per-beat
%                peak amplitude across all detected beats. Cardiac
%                artifact should look consistently large on nearly every
%                beat (low CV); a component that only occasionally spikes
%                near an R-peak by coincidence will have high CV. This is
%                the piece that "score" alone cannot tell you -- score
%                can be inflated by a handful of large coincidental
%                epochs even in a non-cardiac component.
%
% cfg fields:
%   pre_ms, post_ms  - epoch window around each R-peak, ms (default -100, 400;
%                      tighter/faster than blink window since QRS-locked
%                      artifact is briefer than blink morphology)
%   n_perm           - number of random null epochs to compare against (default 200)

if nargin < 4, cfg = struct(); end
if ~isfield(cfg,'pre_ms'),  cfg.pre_ms  = 100; end
if ~isfield(cfg,'post_ms'), cfg.post_ms = 400; end
if ~isfield(cfg,'n_perm'),  cfg.n_perm  = 200; end

pre_n  = round(cfg.pre_ms/1000 * fs);
post_n = round(cfg.post_ms/1000 * fs);
n = length(ic_signal);

valid_events = r_peaks(r_peaks > pre_n & r_peaks < n - post_n);
if isempty(valid_events)
    score = NaN; regularity = NaN;
    diag = struct('n_events', 0);
    return
end

beat_amp = nan(length(valid_events),1);
for i = 1:length(valid_events)
    ep = ic_signal(valid_events(i)-pre_n : valid_events(i)+post_n);
    beat_amp(i) = max(abs(ep - mean(ep)));
end

rng(0); % reproducible null
null_amp = nan(cfg.n_perm,1);
for i = 1:cfg.n_perm
    t = randi([pre_n+1, n-post_n]);
    ep = ic_signal(t-pre_n : t+post_n);
    null_amp(i) = max(abs(ep - mean(ep)));
end

score      = median(beat_amp) / max(median(null_amp), eps);
regularity = std(beat_amp) / max(mean(beat_amp), eps);  % CV; lower = more consistently cardiac-locked

diag = struct('n_events', numel(valid_events), ...
              'beat_amp_median', median(beat_amp), ...
              'null_amp_median', median(null_amp), ...
              'beat_amp_cv', regularity);
end