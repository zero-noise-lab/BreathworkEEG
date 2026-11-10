function [report_txt, zlim, t_vals, masks_pos, masks_neg, eff_size, cluster_p] = ...
    fun_plot_topos_T_stats(data_all, cond_labels, chan_labels, layout, ...
    neighbours, title_str)

% Inputs:
%   data_all    - nchan x nSubj x nCond matrix
%   cond_labels - cell array of condition names
%   chan_labels - cell array of channel labels
%   layout      - fieldtrip layout
%   neighbours  - fieldtrip neighbours struct
%   title_str   - figure title string
%
% Outputs:
%   cluster_p                - struct of cluster-test p-values:
%       .pos_p/.neg_p          1 x (nCond-1), raw cluster p-value for the
%                              largest pos/neg cluster per comparison (NaN
%                              if no cluster of that sign existed)
%       .crit_p_pos/.crit_p_neg  FDR-corrected critical p threshold used
%       .sig_pos/.sig_neg      1 x (nCond-1) logical, whether each
%                              comparison survives FDR correction
%   report_txt             - single multi-line char array with the full
%                            report (same content printed to console),
%                            ready to write directly to a text file
%   zlim, t_vals           - symmetric limits and channel T statistics
%   masks_pos, masks_neg   - FDR-gated positive/negative sensor masks
%   eff_size                - struct of Cohen's dz (paired) effect sizes:
%       .d                   nchan x (nCond-1), every channel's dz
%                            regardless of cluster membership, so any
%                            other summary beyond the fields below can be
%                            pulled from this without rerunning anything.
%       .d_max_pos/.d_mean_pos   1 x (nCond-1), max/mean dz within the
%                            FDR-significant positive cluster (NaN if
%                            none). Reporting both follows Meyer et al.
%                            (2021, Dev Cogn Neurosci) -- max/mean within
%                            the cluster as upper/lower bound on effect size.
%       .d_max_neg/.d_mean_neg  same for the negative cluster; d_max_neg
%                            is the most negative (largest-magnitude
%                            decrease) dz within the cluster.

if nargin < 6, title_str = ''; end

cluster_alpha = 0.05;
test_alpha = 0.05;
numrandomization = 5000;

[nchan, nSubj, nCond_plt] = size(data_all);
nComp = nCond_plt - 1;

% ============================================================
%  CLUSTER STATS: vs first condition, storing T-values + cluster p's
% ============================================================

masks_neg = false(nchan, nComp);
masks_pos = false(nchan, nComp);
t_vals    = nan(nchan, nComp);
effect_d  = nan(nchan, nComp);
pos_p     = nan(1, nComp);
neg_p     = nan(1, nComp);

cfg_stat                  = [];
cfg_stat.method           = 'montecarlo';
cfg_stat.statistic        = 'ft_statfun_depsamplesT';
cfg_stat.correctm         = 'cluster';
cfg_stat.clusteralpha     = cluster_alpha;
cfg_stat.clusterstatistic = 'maxsum';
cfg_stat.tail             = 0;
cfg_stat.clustertail      = 0;
% FieldTrip reports per-tail probabilities by default for tail=0.  Correct
% the cluster probabilities so cfg.alpha=0.05 is a genuine two-sided test.
cfg_stat.correcttail      = 'prob';
cfg_stat.alpha            = test_alpha;
cfg_stat.numrandomization = numrandomization;
if isnumeric(numrandomization) && isscalar(numrandomization) && ...
        2^nSubj <= numrandomization
    cfg_stat.numrandomization = 'all';
end
cfg_stat.neighbours       = neighbours;
cfg_stat.design           = [ones(1,nSubj) 2*ones(1,nSubj); 1:nSubj 1:nSubj];
cfg_stat.ivar             = 1;
cfg_stat.uvar             = 2;

make_tl = @(vec) struct('avg', vec, 'time', 0, 'label', {chan_labels}, 'dimord', 'chan_time');

for ci = 2:nCond_plt
    tl_ref  = arrayfun(@(s) make_tl(data_all(:,s,1)),  1:nSubj, 'UniformOutput', false);
    tl_cond = arrayfun(@(s) make_tl(data_all(:,s,ci)), 1:nSubj, 'UniformOutput', false);

    stat = ft_timelockstatistics(cfg_stat, tl_cond{:}, tl_ref{:});

    t_vals(:,ci-1)    = stat.stat;

    % Cohen's dz (paired), per channel -- matches the paired design of
    % ft_statfun_depsamplesT above. Computed for every channel regardless
    % of cluster membership; d_max_pos/d_mean_pos etc below just index
    % into this using the (FDR-gated) cluster masks.
    diff_vals         = data_all(:,:,ci) - data_all(:,:,1);   % nchan x nSubj
    effect_d(:,ci-1)  = mean(diff_vals, 2, 'omitnan') ./ std(diff_vals, 0, 2, 'omitnan');

    has_pos = isfield(stat, 'posclusters') && ~isempty(stat.posclusters);
    has_neg = isfield(stat, 'negclusters') && ~isempty(stat.negclusters);
    if has_pos
        [pos_p(ci-1), cluster_idx] = min([stat.posclusters.prob]);
        pos_p(ci-1) = local_exact_two_sided_floor(pos_p(ci-1), ...
            cfg_stat.numrandomization, nSubj);
        if isfield(stat, 'posclusterslabelmat')
            labelmat = squeeze(stat.posclusterslabelmat);
            masks_pos(:,ci-1) = labelmat(:) == cluster_idx;
        end
    end
    if has_neg
        [neg_p(ci-1), cluster_idx] = min([stat.negclusters.prob]);
        neg_p(ci-1) = local_exact_two_sided_floor(neg_p(ci-1), ...
            cfg_stat.numrandomization, nSubj);
        if isfield(stat, 'negclusterslabelmat')
            labelmat = squeeze(stat.negclusterslabelmat);
            masks_neg(:,ci-1) = labelmat(:) == cluster_idx;
        end
    end
end

% ============================================================
%  FDR CORRECTION across the nComp comparisons, pos/neg families separate
% ============================================================
[crit_p_pos, sig_pos] = local_fdr_fixed(pos_p,test_alpha);
[crit_p_neg, sig_neg] = local_fdr_fixed(neg_p,test_alpha);

% --- gate plotted markers by the CORRECTED result, not the raw one
for c = 1:nComp
    if ~sig_pos(c), masks_pos(:,c) = false; end
    if ~sig_neg(c), masks_neg(:,c) = false; end
end

% ============================================================
%  EFFECT SIZE SUMMARY: max/mean Cohen's dz within each FDR-significant
%  cluster (Meyer et al., 2021) -- NaN where there's no significant
%  cluster of that sign for that comparison.
% ============================================================
eff_size            = struct();
eff_size.d          = effect_d;
eff_size.d_max_pos  = nan(1, nComp);
eff_size.d_mean_pos = nan(1, nComp);
eff_size.d_max_neg  = nan(1, nComp);
eff_size.d_mean_neg = nan(1, nComp);
for c = 1:nComp
    if any(masks_pos(:,c))
        d_this                 = effect_d(masks_pos(:,c), c);
        eff_size.d_max_pos(c)  = max(d_this);
        eff_size.d_mean_pos(c) = mean(d_this);
    end
    if any(masks_neg(:,c))
        d_this                 = effect_d(masks_neg(:,c), c);
        eff_size.d_max_neg(c)  = min(d_this);   % most negative = largest decrease
        eff_size.d_mean_neg(c) = mean(d_this);
    end
end

% ============================================================
%  REPORTING: build both console output and a returnable text string
% ============================================================
lines = {};
lines{end+1} = sprintf('=== %s: %d fixed contrast-map hypotheses vs %s; strongest cluster per sign; absent cluster p=1 for FDR; effect size = Cohen''s dz (paired), max/mean within the significant cluster (Meyer et al. 2021) ===', title_str, nComp, cond_labels{1});
lines{end+1} = sprintf('%-14s %-10s %-10s %-14s %-9s %-9s %-10s %-10s %-14s %-9s %-9s', ...
    'Comparison', 'pos_p', 'sig(raw)', 'sig(FDR)', 'dz_max+', 'dz_mean+', 'neg_p', 'sig(raw)', 'sig(FDR)', 'dz_max-', 'dz_mean-');
for c = 1:nComp
    raw_pos_sig = ~isnan(pos_p(c)) && pos_p(c) <= test_alpha;
    raw_neg_sig = ~isnan(neg_p(c)) && neg_p(c) <= test_alpha;
    lines{end+1} = sprintf('%-14s %-10.4f %-10s %-14s %-9.3f %-9.3f %-10.4f %-10s %-14s %-9.3f %-9.3f', ...
        sprintf('%s vs %s', cond_labels{c+1}, cond_labels{1}), ...
        pos_p(c), bool2str(raw_pos_sig), bool2str(sig_pos(c)), eff_size.d_max_pos(c), eff_size.d_mean_pos(c), ...
        neg_p(c), bool2str(raw_neg_sig), bool2str(sig_neg(c)), eff_size.d_max_neg(c), eff_size.d_mean_neg(c)); %#ok<AGROW>
end
lines{end+1} = sprintf('FDR critical p: pos=%.4f, neg=%.4f', crit_p_pos, crit_p_neg);

report_txt = sprintf('%s\n', lines{:});   % single multi-line string, one line per cell, newline-terminated
fprintf('\n%s\n', report_txt);            % still print to console as before

% ============================================================
%  PLOT: T-values, shared zlim across all comparisons
% ============================================================

plot_labels = cond_labels(2:end);
nPlots      = nComp;
max_abs_t   = max(abs(t_vals(:)), [], 'omitnan');
if ~isfinite(max_abs_t) || max_abs_t == 0, max_abs_t = 1; end

% ++++ Equate zlim across all plots, -- I first ran the function, got the
% zlims, then set a number that can span all (max of all in same fig.)

% zlim        = [-max_abs_t, max_abs_t];
zlim = [-8.5 8.5];

cfg_lay        = [];
cfg_lay.layout = layout;
lay            = ft_prepare_layout(cfg_lay);

hfigs = gobjects(nPlots, 1);

for c = 1:nPlots
    tl        = [];
    tl.label  = chan_labels(:);
    tl.time   = 0;
    tl.avg    = t_vals(:, c);
    tl.var    = ones(nchan, 1);
    tl.dof    = ones(nchan, 1);
    tl.dimord = 'chan_time';
    tl.cfg    = [];

    cfg                = [];
    cfg.layout         = layout;
    cfg.zlim           = zlim;
    cfg.xlim           = [-0.5 0.5];
    cfg.style          = 'straight';
    cfg.comment        = 'no';
    cfg.colorbar       = 'no';
    cfg.interpolatenan = 'no';

    hfigs(c) = figure('Visible', 'off', 'Renderer', 'painters');
    ft_topoplotER(cfg, tl);

    % negative cluster: filled black circles (only if FDR significant)
    sig_idx = find(masks_neg(:, c));
    if ~isempty(sig_idx)
        [~, lay_idx] = ismember(chan_labels(sig_idx), lay.label);
        lay_idx = lay_idx(lay_idx > 0);
        if ~isempty(lay_idx)
            hold on;
            plot(lay.pos(lay_idx,1), lay.pos(lay_idx,2), ...
                'o', 'MarkerSize', 4, ...
                'MarkerFaceColor', [0.1 0.1 0.1], ...
                'MarkerEdgeColor', [0.1 0.1 0.1], ...
                'LineWidth', 1, 'Tag', 'negative_cluster_marker');
        end
    end

    % positive cluster: white diamonds (only if FDR-corrected significant)
    sig_idx = find(masks_pos(:, c));
    if ~isempty(sig_idx)
        [~, lay_idx] = ismember(chan_labels(sig_idx), lay.label);
        lay_idx = lay_idx(lay_idx > 0);
        if ~isempty(lay_idx)
            hold on;
            plot(lay.pos(lay_idx,1), lay.pos(lay_idx,2), ...
                'o', 'MarkerSize', 4, ...
                'MarkerFaceColor', [1.0 1.0 1.0], ...
                'MarkerEdgeColor', [0.1 0.1 0.1], ...
                'LineWidth', 1, 'Tag', 'positive_cluster_marker');
        end
    end
end

% --- assemble
ncols_fig = min(nPlots, 5);
nrows_fig = ceil(nPlots / ncols_fig);
hout = figure('Position', [100 100 ncols_fig*250 nrows_fig*280], 'Renderer', 'painters');
colormap(hout, parula(256));

for c = 1:nPlots
    src_ax = findall(hfigs(c), 'type', 'axes');
    dst_ax = subplot(nrows_fig, ncols_fig, c, 'Parent', hout);
    copyobj(src_ax.Children, dst_ax);
    dst_ax.CLim               = src_ax.CLim;
    dst_ax.DataAspectRatio    = src_ax.DataAspectRatio;
    dst_ax.XLim               = src_ax.XLim;
    dst_ax.YLim               = src_ax.YLim;
    dst_ax.PlotBoxAspectRatio = src_ax.PlotBoxAspectRatio;
    axis(dst_ax, 'off');

    sig_str = '';
    if any(masks_neg(:,c)), sig_str = [sig_str ' \downarrow']; end
    if any(masks_pos(:,c)), sig_str = [sig_str ' \uparrow']; end
    title(dst_ax, [plot_labels{c} sig_str]);
end

close(hfigs);

cb = colorbar('Position', [0.93 0.2 0.02 0.6]);
ylabel(cb, 'T-value');
set(cb,'FontSize', 12);

ax_all = findall(hout, 'type', 'axes');
for a = 1:numel(ax_all)
    pos = ax_all(a).Position;
    ax_all(a).Position = [pos(1) pos(2) pos(3)*0.88 pos(4)];
end

sgtitle(sprintf('%s: T-value vs %s (n=%d, FDR-corrected) | o=decrease  d=increase', ...
    title_str, cond_labels{1}, nSubj), 'FontSize', 13, 'FontWeight', 'bold');

cluster_p            = struct();
cluster_p.pos_p      = pos_p;
cluster_p.neg_p      = neg_p;
cluster_p.crit_p_pos = crit_p_pos;
cluster_p.crit_p_neg = crit_p_neg;
cluster_p.sig_pos    = sig_pos;
cluster_p.sig_neg    = sig_neg;

end

% -------------------------------------------------------------------
function [crit_p, sig] = local_fdr_fixed(p,alpha)
% Every requested contrast is a member of the family. No cluster = p=1.
valid = isfinite(p);
p_for_fdr = p;
p_for_fdr(~valid) = 1;
p_sorted = sort(p_for_fdr);
m = numel(p_sorted);
below = p_sorted <= (1:m)/m * alpha;
if any(below), crit_p = p_sorted(find(below,1,'last')); else, crit_p = 0; end
sig = valid & (p_for_fdr <= crit_p);
end

function p = local_exact_two_sided_floor(p,numrandomization,nSubj)
% FieldTrip's cluster probability uses a strict exceedance count and can
% therefore return zero when every 2^n sign flip is enumerated.  For a
% two-sided exhaustive randomization test, the observed allocation and its
% sign-reversed counterpart imply a minimum attainable probability 2/2^n.
% Enforce only that exact-test support bound; Monte Carlo results are left
% unchanged because FieldTrip already reports their sampled probability.
if ischar(numrandomization) && strcmpi(numrandomization,'all') && isfinite(p)
    p = max(p, min(1, 2 / (2^nSubj)));
end
end

function s = bool2str(b)
if b, s = 'yes'; else, s = 'no'; end
end