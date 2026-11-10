function [report_txt, t_topo_bands, stat_bands, eff_size, cluster_p] = fun_plot_topos_iv_regression_bands(data_all, iv_all, chan_labels, layout, neighbours, band_labels, iv_label)
% FUN_PLOT_TOPOS_IV_REGRESSION_BANDS  Within-subject regression (topo vs
% continuous IV, e.g. ASC rating) run separately per band, plotted
% together in ONE figure with a SHARED zlim, with FDR correction applied
% ACROSS the bands tested in this call (positive and negative clusters
% corrected separately). Markers reflect the FDR-CORRECTED result, not
% each band's raw within-test significance.
%
% Effect size: per-channel within-subject Pearson correlation r between
% the topo measure and the IV, after removing each subject's own mean
% from both (so only within-subject covariation contributes, matching
% what depsamplesregrT actually tests) -- same definition as the
% non-banded fun_plot_topos_iv_regression. Summarized as max/mean r
% within each FDR-significant cluster per band (Meyer et al., 2021).
%
% Usage:
%   [report_txt, t_topo_bands, stat_bands, eff_size, cluster_p] = ...
%       fun_plot_topos_iv_regression_bands(band_topos, asc_meas_iv, chan_labels_63, layout_63, neighbours_63, ...
%       {bands.name}, 'ASC rating');
%
% Inputs:
%   data_all    - nchan x nSubj x nCond x nBand
%   iv_all      - nSubj x nCond (same IV across all bands, e.g. ASC rating)
%   chan_labels, layout, neighbours - as in fun_plot_topos_iv_regression
%   band_labels - cell array of band names, length nBand
%   iv_label    - name of the IV, for titles/labels
%
% Outputs:
%   report_txt    - single multi-line string, ready to write to a text file
%   t_topo_bands  - 1 x nBand cell array, each cell the T-value vector (nchan x 1)
%   stat_bands    - 1 x nBand cell array with the full stat output per band
%   eff_size      - struct: .r (nchan x nBand, per-channel within-subject
%                   correlation), .r_max_pos/.r_mean_pos/.r_max_neg/.r_mean_neg
%                   (each 1 x nBand, NaN where that band has no FDR-significant
%                   cluster of that sign)
%   cluster_p     - struct: .pos_p/.neg_p (1 x nBand raw cluster p, NaN if
%                   no cluster), .crit_p_pos/.crit_p_neg (FDR-corrected
%                   critical p threshold used), .sig_pos/.sig_neg (1 x nBand
%                   logical, survives FDR correction across bands)

[nchan, nSubj, nCond, nBand] = size(data_all);
assert(numel(band_labels) == nBand, 'band_labels length (%d) must match nBand (%d).', numel(band_labels), nBand);

make_tl = @(vec) struct('avg', vec, 'time', 0, 'label', {chan_labels}, 'dimord', 'chan_time');

t_topo_bands   = cell(1, nBand);
mask_neg_bands = false(nchan, nBand);
mask_pos_bands = false(nchan, nBand);
stat_bands     = cell(1, nBand);
pos_p          = nan(1, nBand);
neg_p          = nan(1, nBand);
effect_r       = nan(nchan, nBand);

% ============================================================
%  STATS: run the within-subject regression once per band
% ============================================================
for b = 1:nBand
    tl_all    = cell(1, nSubj*nCond);
    data_long = nan(nchan, nSubj*nCond);
    iv_vec    = nan(1, nSubj*nCond);
    subj_vec  = nan(1, nSubj*nCond);
    idx = 0;
    for s = 1:nSubj
        for c = 1:nCond
            idx = idx + 1;
            tl_all{idx}      = make_tl(data_all(:,s,c,b));
            data_long(:,idx) = data_all(:,s,c,b);
            iv_vec(idx)      = iv_all(s,c);
            subj_vec(idx)    = s;
        end
    end

    valid     = ~isnan(iv_vec);
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
    % Convert FieldTrip's default per-tail probabilities to two-sided
    % probabilities before thresholding and across-band FDR correction.
    cfg_stat.correcttail      = 'prob';
    cfg_stat.alpha            = 0.05;
    cfg_stat.numrandomization = 5000;
    cfg_stat.neighbours       = neighbours;
    cfg_stat.design           = [iv_vec; subj_vec];
    cfg_stat.ivar             = 1;
    cfg_stat.uvar             = 2;

    stat = ft_timelockstatistics(cfg_stat, tl_all{:});

    t_topo_bands{b} = stat.stat(:);
    stat_bands{b}   = stat;

    has_pos = isfield(stat, 'posclusters') && ~isempty(stat.posclusters);
    has_neg = isfield(stat, 'negclusters') && ~isempty(stat.negclusters);
    if has_pos
        [pos_p(b), cluster_idx] = min([stat.posclusters.prob]);
        if isfield(stat, 'posclusterslabelmat')
            labelmat = squeeze(stat.posclusterslabelmat);
            mask_pos_bands(:,b) = labelmat(:) == cluster_idx;
        end
    end
    if has_neg
        [neg_p(b), cluster_idx] = min([stat.negclusters.prob]);
        if isfield(stat, 'negclusterslabelmat')
            labelmat = squeeze(stat.negclusterslabelmat);
            mask_neg_bands(:,b) = labelmat(:) == cluster_idx;
        end
    end

    % --- effect size: per-channel within-subject correlation, this band
    data_dm = data_long;
    iv_dm   = iv_vec;
    for s = unique(subj_vec)
        m            = subj_vec == s;
        data_dm(:,m) = data_dm(:,m) - mean(data_dm(:,m), 2);
        iv_dm(m)     = iv_dm(m) - mean(iv_dm(m));
    end
    effect_r(:,b) = sum(data_dm .* iv_dm, 2) ./ sqrt(sum(data_dm.^2, 2) .* sum(iv_dm.^2));
end

% ============================================================
%  FDR CORRECTION across the nBand tests, pos/neg families separate
% ============================================================
[crit_p_pos, sig_pos] = local_fdr_fixed(pos_p);
[crit_p_neg, sig_neg] = local_fdr_fixed(neg_p);

% --- gate plotted markers by the CORRECTED result, not the raw one
for b = 1:nBand
    if ~sig_pos(b), mask_pos_bands(:,b) = false; end
    if ~sig_neg(b), mask_neg_bands(:,b) = false; end
end

% ============================================================
%  EFFECT SIZE SUMMARY: max/mean r within each FDR-significant cluster
% ============================================================
eff_size            = struct();
eff_size.r          = effect_r;
eff_size.r_max_pos  = nan(1, nBand);
eff_size.r_mean_pos = nan(1, nBand);
eff_size.r_max_neg  = nan(1, nBand);
eff_size.r_mean_neg = nan(1, nBand);
for b = 1:nBand
    if any(mask_pos_bands(:,b))
        r_this                 = effect_r(mask_pos_bands(:,b), b);
        eff_size.r_max_pos(b)  = max(r_this);
        eff_size.r_mean_pos(b) = mean(r_this);
    end
    if any(mask_neg_bands(:,b))
        r_this                 = effect_r(mask_neg_bands(:,b), b);
        eff_size.r_max_neg(b)  = min(r_this);   % most negative = largest decrease
        eff_size.r_mean_neg(b) = mean(r_this);
    end
end

cluster_p            = struct();
cluster_p.pos_p      = pos_p;
cluster_p.neg_p      = neg_p;
cluster_p.crit_p_pos = crit_p_pos;
cluster_p.crit_p_neg = crit_p_neg;
cluster_p.sig_pos    = sig_pos;
cluster_p.sig_neg    = sig_neg;

% ============================================================
%  REPORTING
% ============================================================
lines = {};
lines{end+1} = sprintf('=== %s ~ %s: %d fixed band-map hypotheses; strongest cluster per sign; absent cluster p=1 for FDR ===', ' ', iv_label, nBand);
lines{end+1} = sprintf('%-14s %-10s %-10s %-14s %-8s %-8s %-10s %-10s %-14s %-8s %-8s', ...
    'Band', 'pos_p', 'sig(raw)', 'sig(FDR)', 'r_max+', 'r_mean+', 'neg_p', 'sig(raw)', 'sig(FDR)', 'r_max-', 'r_mean-');
for b = 1:nBand
    raw_pos_sig = ~isnan(pos_p(b)) && pos_p(b) <= 0.05;
    raw_neg_sig = ~isnan(neg_p(b)) && neg_p(b) <= 0.05;
    lines{end+1} = sprintf('%-14s %-10.4f %-10s %-14s %-8.3f %-8.3f %-10.4f %-10s %-14s %-8.3f %-8.3f', band_labels{b}, ...
        pos_p(b), bool2str(raw_pos_sig), bool2str(sig_pos(b)), eff_size.r_max_pos(b), eff_size.r_mean_pos(b), ...
        neg_p(b), bool2str(raw_neg_sig), bool2str(sig_neg(b)), eff_size.r_max_neg(b), eff_size.r_mean_neg(b)); %#ok<AGROW>
end
lines{end+1} = sprintf('FDR critical p: pos=%.4f, neg=%.4f', crit_p_pos, crit_p_neg);

report_txt = sprintf('%s\n', lines{:});
fprintf('\n%s\n', report_txt);

% ============================================================
%  SHARED zlim, computed across ALL bands jointly
% ============================================================
all_t = cat(1, t_topo_bands{:});
shared_zlim = [-max(abs(all_t)), max(abs(all_t))];

% ============================================================
%  PLOT: one panel per band, same zlim, significance markers (FDR-corrected)
% ============================================================
cfg_lay        = [];
cfg_lay.layout = layout;
lay            = ft_prepare_layout(cfg_lay);

hfigs = gobjects(nBand,1);
for b = 1:nBand
    tl        = [];
    tl.label  = chan_labels(:);
    tl.time   = 0;
    tl.avg    = t_topo_bands{b};
    tl.var    = ones(nchan,1);
    tl.dof    = ones(nchan,1);
    tl.dimord = 'chan_time';
    tl.cfg    = [];

    cfg          = [];
    cfg.layout   = layout;
    cfg.zlim     = shared_zlim;
    cfg.xlim     = [-0.5 0.5];
    cfg.style    = 'straight';
    cfg.comment  = 'no';
    cfg.colorbar = 'no';
    cfg.interpolatenan = 'no';

    hfigs(b) = figure('Visible', 'off');
    ft_topoplotER(cfg, tl);

    sig_idx = find(mask_neg_bands(:,b));
    if ~isempty(sig_idx)
        [~, lay_idx] = ismember(chan_labels(sig_idx), lay.label);
        lay_idx = lay_idx(lay_idx > 0);
        if ~isempty(lay_idx)
            hold on;
            plot(lay.pos(lay_idx,1), lay.pos(lay_idx,2), 'o', 'MarkerSize', 6, ...
                'MarkerFaceColor', [0.1 0.1 0.1], 'MarkerEdgeColor', [0.1 0.1 0.1], ...
                'LineWidth', 1.2, 'Tag', 'negative_cluster_marker', 'UserData', b);
        end
    end
    sig_idx = find(mask_pos_bands(:,b));
    if ~isempty(sig_idx)
        [~, lay_idx] = ismember(chan_labels(sig_idx), lay.label);
        lay_idx = lay_idx(lay_idx > 0);
        if ~isempty(lay_idx)
            hold on;
            plot(lay.pos(lay_idx,1), lay.pos(lay_idx,2), 'd', 'MarkerSize', 6, ...
                'MarkerFaceColor', [1 1 1], 'MarkerEdgeColor', [0.1 0.1 0.1], ...
                'LineWidth', 1.2, 'Tag', 'positive_cluster_marker', 'UserData', b);
        end
    end
end

ncols_fig = min(nBand, 5);
nrows_fig = ceil(nBand / ncols_fig);
hout = figure('Position', [100 100 ncols_fig*250 nrows_fig*280]);

for b = 1:nBand
    src_ax = findall(hfigs(b), 'type', 'axes');
    dst_ax = subplot(nrows_fig, ncols_fig, b, 'Parent', hout);
    copyobj(src_ax.Children, dst_ax);
    dst_ax.CLim               = src_ax.CLim;
    dst_ax.DataAspectRatio    = src_ax.DataAspectRatio;
    dst_ax.XLim               = src_ax.XLim;
    dst_ax.YLim               = src_ax.YLim;
    dst_ax.PlotBoxAspectRatio = src_ax.PlotBoxAspectRatio;
    axis(dst_ax, 'off');

    sig_str = '';
    if any(mask_neg_bands(:,b)), sig_str = [sig_str ' \downarrow']; end
    if any(mask_pos_bands(:,b)), sig_str = [sig_str ' \uparrow']; end
    title_handle=title(dst_ax, [band_labels{b} sig_str]);
    title_handle.Tag='band_cluster_title';
    title_handle.UserData=b;
end
close(hfigs);

cb = colorbar('Position', [0.93 0.2 0.02 0.6]);
set(cb,'FontSize', 12);
sgtitle(sprintf('T-value: topo ~ %s (within-subject, n=%d, FDR-corrected) | shared zlim | o=negative T  d=positive T', ...
    iv_label, nSubj), 'FontSize', 13, 'FontWeight', 'bold');

end

% -------------------------------------------------------------------
function [crit_p, sig] = local_fdr_fixed(p)
% Every requested band is a family member. No cluster = p=1.
valid = isfinite(p);
p_for_fdr = p;
p_for_fdr(~valid) = 1;
p_sorted = sort(p_for_fdr);
m = numel(p_sorted);
below = p_sorted <= (1:m)/m * 0.05;
if any(below), crit_p = p_sorted(find(below,1,'last')); else, crit_p = 0; end
sig = valid & (p_for_fdr <= crit_p);
end

function s = bool2str(b)
if b, s = 'yes'; else, s = 'no'; end
end