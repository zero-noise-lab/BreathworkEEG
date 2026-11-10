
function [report_txt, t_topo_out, eff_size, cluster_p] = fun_plot_topos_iv_regression(data_all, iv_all, chan_labels, layout, neighbours, title_str, iv_label)
% FUN_PLOT_TOPOS_IV_REGRESSION  Within-subject cluster-based regression of
% topographic data against a continuous independent variable (e.g. ASC
% hand-sign rating). Pools all subject*condition observations into ONE
% regression (ft_statfun_depsamplesregrT), with subject as the
% repeated-measures blocking variable -- tests whether the topo measure
% covaries with the IV as it varies WITHIN each subject across
% conditions. Produces ONE topography (the regression T-map).
%
% Effect size: per-channel within-subject Pearson correlation r between
% the topo measure and the IV, after removing each subject's own mean
% from both (so only within-subject covariation contributes, matching
% what depsamplesregrT actually tests). Summarized as max/mean r within
% each significant cluster, same max+mean-in-cluster logic used for the
% T-stats topos (Meyer et al., 2021).
%
% NOTE: the 'per_condition' mode (separate between-subjects regression
% per condition) has been removed -- this now matches
% fun_plot_topos_iv_regression_bands, which only ever ran the
% within-subject version. If you need per-condition regression again
% later, that logic is still recoverable from earlier versions of this
% function.
%
% Usage:
%   [report_txt, t_topo_out, eff_size, cluster_p] = ...
%       fun_plot_topos_iv_regression(exponent_all, asc_all, chan_labels, layout, neighbours, 'Exponent vs ASC', 'ASC rating');
%
% Inputs:
%   data_all    - nchan x nSubj x nCond matrix (e.g. exponent, band power)
%   iv_all      - nSubj x nCond matrix of the continuous IV
%   chan_labels, layout, neighbours - as before
%   title_str   - figure title string
%   iv_label    - name of the IV, used in axis/colorbar labels
%
% Outputs:
%   report_txt  - single multi-line string, ready to write to a text file
%   t_topo_out  - the timelock struct with the regression T-map (tl.avg = T-values)
%   eff_size    - struct: .r (per-channel within-subject correlation),
%                 .r_max_pos/.r_mean_pos, .r_max_neg/.r_mean_neg (NaN if
%                 that cluster doesn't exist)
%   cluster_p   - struct: .pos_p/.neg_p (cluster p-value, NaN if none),
%                 .sig_pos/.sig_neg (pos_p/neg_p <= 0.05 -- plain
%                 threshold, no FDR here, since this is a single test
%                 with no multi-comparison family to correct against;
%                 unlike the _bands version, which should be FDR-corrected
%                 across bands)

[nchan, nSubj, nCond] = size(data_all);
assert(isequal(size(iv_all), [nSubj, nCond]), ...
    'iv_all must be nSubj x nCond (%d x %d) to match data_all -- got %s.', ...
    nSubj, nCond, mat2str(size(iv_all)));

make_tl = @(vec) struct('avg', vec, 'time', 0, 'label', {chan_labels}, 'dimord', 'chan_time');

cfg_lay        = [];
cfg_lay.layout = layout;
lay            = ft_prepare_layout(cfg_lay);

% ============================================================
%  POOL ALL SUBJECT x CONDITION OBSERVATIONS INTO ONE REGRESSION
% ============================================================
tl_all    = cell(1, nSubj*nCond);
data_long = nan(nchan, nSubj*nCond);
iv_vec    = nan(1, nSubj*nCond);
subj_vec  = nan(1, nSubj*nCond);
idx = 0;
for s = 1:nSubj
    for c = 1:nCond
        idx = idx + 1;
        tl_all{idx}      = make_tl(data_all(:,s,c));
        data_long(:,idx) = data_all(:,s,c);
        iv_vec(idx)      = iv_all(s,c);
        subj_vec(idx)    = s;
    end
end

valid = ~isnan(iv_vec);   % drop subject*condition pairs missing an IV rating
if any(~valid)
    fprintf('Dropping %d/%d subject*condition observations with missing IV.\n', sum(~valid), numel(valid));
end
tl_all    = tl_all(valid);
data_long = data_long(:,valid);
iv_vec    = iv_vec(valid);
subj_vec  = subj_vec(valid);

cfg_stat                  = [];
cfg_stat.method           = 'montecarlo';
cfg_stat.statistic        = 'ft_statfun_depsamplesregrT';
cfg_stat.correctm         = 'cluster';
cfg_stat.clusteralpha     = 0.05;
cfg_stat.clusterstatistic = 'maxsum';
cfg_stat.tail             = 0;
cfg_stat.clustertail      = 0;
% Use two-sided cluster probabilities (FieldTrip otherwise returns
% one-sided probabilities separately for each tail when tail=0).
cfg_stat.correcttail      = 'prob';
cfg_stat.alpha            = 0.05;
cfg_stat.numrandomization = 5000;
cfg_stat.neighbours       = neighbours;
cfg_stat.design           = [iv_vec; subj_vec];
cfg_stat.ivar             = 1;
cfg_stat.uvar             = 2;

stat = ft_timelockstatistics(cfg_stat, tl_all{:});

has_pos = isfield(stat, 'posclusters') && ~isempty(stat.posclusters);
has_neg = isfield(stat, 'negclusters') && ~isempty(stat.negclusters);
pos_p = nan; neg_p = nan;
mask_pos = false(nchan,1);
mask_neg = false(nchan,1);
if has_pos
    [pos_p, cluster_idx] = min([stat.posclusters.prob]);
    if isfield(stat,'posclusterslabelmat')
        labelmat = squeeze(stat.posclusterslabelmat);
        mask_pos = labelmat(:) == cluster_idx;
    end
end
if has_neg
    [neg_p, cluster_idx] = min([stat.negclusters.prob]);
    if isfield(stat,'negclusterslabelmat')
        labelmat = squeeze(stat.negclusterslabelmat);
        mask_neg = labelmat(:) == cluster_idx;
    end
end
sig_pos = ~isnan(pos_p) && pos_p <= 0.05;
sig_neg = ~isnan(neg_p) && neg_p <= 0.05;
if ~sig_pos, mask_pos(:) = false; end
if ~sig_neg, mask_neg(:) = false; end

% ============================================================
%  EFFECT SIZE: per-channel within-subject correlation
% ============================================================
data_dm = data_long;
iv_dm   = iv_vec;
for s = unique(subj_vec)
    m           = subj_vec == s;
    data_dm(:,m) = data_dm(:,m) - mean(data_dm(:,m), 2);
    iv_dm(m)     = iv_dm(m) - mean(iv_dm(m));
end
effect_r = sum(data_dm .* iv_dm, 2) ./ sqrt(sum(data_dm.^2, 2) .* sum(iv_dm.^2));

eff_size            = struct();
eff_size.r          = effect_r;
eff_size.r_max_pos  = nan;
eff_size.r_mean_pos = nan;
eff_size.r_max_neg  = nan;
eff_size.r_mean_neg = nan;
if any(mask_pos)
    eff_size.r_max_pos  = max(effect_r(mask_pos));
    eff_size.r_mean_pos = mean(effect_r(mask_pos));
end
if any(mask_neg)
    eff_size.r_max_neg  = min(effect_r(mask_neg));   % most negative = largest decrease
    eff_size.r_mean_neg = mean(effect_r(mask_neg));
end

cluster_p         = struct();
cluster_p.pos_p   = pos_p;
cluster_p.neg_p   = neg_p;
cluster_p.sig_pos = sig_pos;
cluster_p.sig_neg = sig_neg;

% ============================================================
%  REPORTING
% ============================================================
lines = {};
lines{end+1} = sprintf('=== %s ~ %s (within-subject, n=%d) ===', title_str, iv_label, nSubj);
lines{end+1} = sprintf('pos_p = %.4f (sig: %s)  r_max+ = %.3f  r_mean+ = %.3f', ...
    pos_p, bool2str(sig_pos), eff_size.r_max_pos, eff_size.r_mean_pos);
lines{end+1} = sprintf('neg_p = %.4f (sig: %s)  r_max- = %.3f  r_mean- = %.3f', ...
    neg_p, bool2str(sig_neg), eff_size.r_max_neg, eff_size.r_mean_neg);
report_txt = sprintf('%s\n', lines{:});
fprintf('\n%s\n', report_txt);

% ============================================================
%  PLOT: single topography, the regression T-map
% ============================================================
hfig = figure('Position', [100 100 400 420]); %#ok<NASGU>
tl        = [];
tl.label  = chan_labels(:);
tl.time   = 0;
tl.avg    = stat.stat;
tl.var    = ones(nchan,1);
tl.dof    = ones(nchan,1);
tl.dimord = 'chan_time';
tl.cfg    = [];

cfg          = [];
cfg.layout   = layout;
cfg.xlim     = [-0.5 0.5];
cfg.style    = 'straight';
cfg.comment  = 'no';
cfg.colorbar = 'no';
cfg.interpolatenan = 'no';

ft_topoplotER(cfg, tl);
t_topo_out = tl;

sig_idx = find(mask_neg);
if ~isempty(sig_idx)
    [~, lay_idx] = ismember(chan_labels(sig_idx), lay.label);
    lay_idx = lay_idx(lay_idx > 0);
    if ~isempty(lay_idx)
        hold on;
        plot(lay.pos(lay_idx,1), lay.pos(lay_idx,2), 'o', 'MarkerSize', 6, ...
            'MarkerFaceColor', [0.1 0.1 0.1], 'MarkerEdgeColor', [0.1 0.1 0.1], ...
            'LineWidth', 1.2, 'Tag', 'negative_cluster_marker');
    end
end
sig_idx = find(mask_pos);
if ~isempty(sig_idx)
    [~, lay_idx] = ismember(chan_labels(sig_idx), lay.label);
    lay_idx = lay_idx(lay_idx > 0);
    if ~isempty(lay_idx)
        hold on;
        plot(lay.pos(lay_idx,1), lay.pos(lay_idx,2), 'd', 'MarkerSize', 6, ...
            'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0.1 0.1 0.1], ...
            'LineWidth', 1.2, 'Tag', 'positive_cluster_marker');
    end
end

cb = colorbar('Position', [0.88 0.2 0.03 0.6]);
set(cb,'FontSize', 12);
sig_str = '';
if any(mask_neg), sig_str = [sig_str ' \downarrow']; end
if any(mask_pos), sig_str = [sig_str ' \uparrow']; end
sgtitle(sprintf('%s ~ %s (within-subject, n=%d)%s | o=negative T  d=positive T', ...
    title_str, iv_label, nSubj, sig_str), 'FontSize', 13, 'FontWeight', 'bold');

end

function s = bool2str(b)
if b, s = 'yes'; else, s = 'no'; end
end