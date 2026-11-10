function [score, diag] = blink_locked_score(ic_signal, blink_events, fs, cfg)
% BLINK_LOCKED_SCORE  Quantify whether an IC's amplitude is specifically
% elevated at blink timestamps, vs. random timestamps.
%
%   [score, diag] = blink_locked_score(ic_signal, blink_events, fs, cfg)
%
% score = (median abs amplitude in blink-locked epochs) /
%         (median abs amplitude in null-distribution of shuffled epochs)
% Values well above 1 indicate the component's amplitude tracks blink
% timing specifically, not just a shared slow rhythm (e.g. alpha) that
% would produce similar amplitude everywhere regardless of blink timing.
%
% cfg fields:
%   pre_ms, post_ms  - epoch window around each blink, ms (default -200, 400)
%   n_perm           - number of random null epochs to compare against (default 200)

if nargin < 4, cfg = struct(); end
if ~isfield(cfg,'pre_ms'),  cfg.pre_ms  = 200; end
if ~isfield(cfg,'post_ms'), cfg.post_ms = 400; end
if ~isfield(cfg,'n_perm'),  cfg.n_perm  = 200; end

pre_n  = round(cfg.pre_ms/1000 * fs);
post_n = round(cfg.post_ms/1000 * fs);
n = length(ic_signal);

valid_events = blink_events(blink_events > pre_n & blink_events < n - post_n);
if isempty(valid_events)
    score = NaN;
    diag = struct('n_events', 0, 'blink_amp_median', NaN, 'null_amp_median', NaN);
    return
end

blink_amp = nan(length(valid_events),1);
for i = 1:length(valid_events)
    ep = ic_signal(valid_events(i)-pre_n : valid_events(i)+post_n);
    blink_amp(i) = max(abs(ep - mean(ep)));
end

rng(0); % reproducible null distribution
null_amp = nan(cfg.n_perm,1);
for i = 1:cfg.n_perm
    t = randi([pre_n+1, n-post_n]);
    ep = ic_signal(t-pre_n : t+post_n);
    null_amp(i) = max(abs(ep - mean(ep)));
end

score = median(blink_amp) / max(median(null_amp), eps);
diag = struct('n_events', numel(valid_events), ...
              'blink_amp_median', median(blink_amp), ...
              'null_amp_median',  median(null_amp));
end