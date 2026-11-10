
function [data_cln, ep_info] = fun_epoch_reject(data_seg, epochparams, reject_params)
% FUN_EPOCH_REJECT  Epoch data and reject bad epochs by threshold criteria
%
% Usage:
%   [data_cln, ep_info] = fun_epoch_reject(data_seg, epochparams, reject_params)
%
% Inputs:
%   data_seg       - FT continuous data (1 trial per condition)
%   epochparams    - struct with .length (sec) and .overlap (0-1)
%   reject_params  - struct with rejection thresholds:
%     .abs_thresh    - absolute voltage threshold (uV), default 150
%     .p2p_thresh    - peak-to-peak threshold (uV), default 200
%     .epoch_std     - epoch std threshold (uV), default 50
%     .var_ratio     - epoch variance / median variance, default 5
%
% Outputs:
%   data_cln  - FT data with only clean epochs as trials
%   ep_info   - struct with rejection diagnostics


% --- step 1: epoch via fieldtrip
cfg         = [];
cfg.length  = epochparams.length;
cfg.overlap = epochparams.overlap;
data_ep     = ft_redefinetrial(cfg, data_seg);

nEpochs = numel(data_ep.trial);
nchan   = numel(data_ep.label);

% --- step 2: skip NaN epochs first
has_nan = false(nEpochs, 1);
for ep = 1:nEpochs
    has_nan(ep) = any(isnan(data_ep.trial{ep}(:)));
end

% --- step 3: compute per-epoch metrics on non-NaN epochs
is_bad     = has_nan;   % NaN epochs already marked
epoch_var  = zeros(nEpochs, 1);   % mean variance across channels

for ep = 1:nEpochs
    
    if has_nan(ep), continue; end
    
    dat = data_ep.trial{ep};   % nchan x nSamples
    
    if ~isnan(reject_params.abs_thresh)
        % absolute threshold
        if any(abs(dat(:)) > reject_params.abs_thresh)
            is_bad(ep) = true;
            continue
        end
    end
    
    if ~isnan(reject_params.p2p_thresh)
        % peak-to-peak
        p2p = max(dat, [], 2) - min(dat, [], 2);
        if any(p2p > reject_params.p2p_thresh)
            is_bad(ep) = true;
            continue
        end
    end
    
    if ~isnan(reject_params.p2p_thresh)
        % epoch std
        ep_std = std(dat, 0, 2);
        if any(ep_std > reject_params.epoch_std)
            is_bad(ep) = true;
            continue
        end
        
    end
    epoch_var(ep) = mean(var(dat, 0, 2));
    
end

% --- step 4: variance ratio check on surviving epochs
clean_idx  = find(~is_bad);
if ~isempty(clean_idx)
    median_var = median(epoch_var(clean_idx));
    for ep = clean_idx'
        if epoch_var(ep) > median_var * reject_params.var_ratio
            is_bad(ep) = true;
        end
    end
end

% --- step 5: select clean epochs
good_idx = find(~is_bad);

cfg        = [];
cfg.trials = good_idx;
data_cln   = ft_selectdata(cfg, data_ep);

% --- diagnostics
ep_info.n_total      = nEpochs;
ep_info.n_nan        = sum(has_nan);
ep_info.n_rejected   = sum(is_bad & ~has_nan);
ep_info.n_clean      = numel(good_idx);
ep_info.pct_clean    = 100 * numel(good_idx) / nEpochs;
ep_info.reject_params = reject_params;
ep_info.good_idx     = good_idx;

fprintf('  Epochs: %d total | %d NaN | %d rejected | %d clean (%.0f%%)\n', ...
    ep_info.n_total, ep_info.n_nan, ep_info.n_rejected, ...
    ep_info.n_clean, ep_info.pct_clean);

end