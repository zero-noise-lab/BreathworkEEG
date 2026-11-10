function [bad_idx, chan_std, threshold] = check_channel_quality_wholetrial(data_segment, std_mult, center_fun)
% Flag channels whose STD over an ENTIRE TRIAL is higher than mean / median
% ofl other channelss

% INPUTS
%   data_segment - channels x samples, one trial
%   std_mult     - multiplier on the reference channel-STD, default 2.5
%   center_fun   - 'median' (default) or 'mean' 
%
% OUTPUTS
%   bad_idx   - indices of channels flagged bad for this trial
%   chan_std  - STD per channel over the full trial 
%   threshold - the STD value channels had to exceed to be flagged...


if nargin < 2 || isempty(std_mult),   std_mult   = 2.5;      end
if nargin < 3 || isempty(center_fun), center_fun = 'median'; end

chan_std = std(double(data_segment), 0, 2);  % n_chan x 1

switch lower(center_fun)
    case 'median'
        center = median(chan_std);
    case 'mean'
        center = mean(chan_std);
    otherwise
        error('center_fun must be ''median'' or ''mean''.');
end

threshold = std_mult * center;
bad_idx = find(chan_std > threshold);

end