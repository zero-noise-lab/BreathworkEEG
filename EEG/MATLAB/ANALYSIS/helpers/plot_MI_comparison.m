function [report_txt, stats_own, stats_direct, MI_all] = plot_MI_comparison(periodic_ga, aperiodic_ga, total_ga, cond_labels_full, bands, titletext)
% PLOT_MI_COMPARISON  Same 3-measure overlay structure as
% plot_pct_change_comparison, but using the bounded modulation index
% MI = (condition - baseline) / (|condition| + |baseline|) instead of
% percent change, computed separately for periodic, aperiodic, and total.
% Absolute values in the denominator guarantee MI stays within [-1, 1]
% regardless of sign (triangle inequality), which plain (condition +
% baseline) does not -- necessary because periodic power (band_fooof_auc)
% can be negative and was observed to cross zero in gamma.
%
% Runs TWO kinds of tests, both with rank-biserial correlation effect
% sizes (Kerby, 2014):
%   1) For EACH measure separately: does its own MI-vs-baseline differ
%      from zero at R1-R4 -- small color-matched marker on that
%      measure's own line.
%   2) DIRECT comparison: does periodic's MI differ from aperiodic's MI
%      (paired, per condition) -- prominent black star. Total is NOT
%      part of this direct test, shown for visual context only.
%
% All FDR-gated WITHIN each band, across that band's 4 R-vs-baseline
% comparisons -- each of the 4 test families corrected separately.
%
% report_txt is a readable stats report (one section per band, all four
% test families) -- write it to a file yourself, e.g.:
%   fid = fopen(fullfile(RESULTS_DIR, 'exponent_MI_stats.txt'), 'w');
%   fprintf(fid, '%s', report_txt);
%   fclose(fid);
%
% Usage:
%   [report_txt, stats_own, stats_direct, MI_all] = plot_MI_comparison( ...
%       fooof_auc_ga, aper_auc_ga, total_auc_ga, cond_labels_full, bands, ...
%       'Modulation index vs baseline');
%
% Inputs:
%   periodic_ga, aperiodic_ga, total_ga - each nSubj x nCondFull x nBand,
%                                          RAW units, baseline in column 1
%   cond_labels_full - {'baseline','R1','R2','R3','R4'}
%   bands            - bands struct with .name
%   titletext        - figure title
%
% Outputs:
%   report_txt   - the stats report as one char array
%   stats_own    - struct .p/.sig/.eff, each itself a struct with
%                  .periodic/.aperiodic/.total (nBand x nCondDiff) --
%                  each measure's own MI tested against zero
%   stats_direct - struct .p/.sig/.eff (nBand x nCondDiff) -- periodic
%                  vs aperiodic, direct paired comparison
%   MI_all       - struct .periodic/.aperiodic/.total, each
%                  nSubj x nCondFull x nBand

nBand     = size(bands,2);
nCondFull = size(cond_labels_full, 2);
nCondDiff = nCondFull - 1;

measure_colors = [0.85 0.10 0.10;   % periodic - red
                   0.10 0.30 0.85;   % aperiodic - blue
                   0.40 0.40 0.40];  % total - grey

MI_periodic  = nan(size(periodic_ga,1), nCondFull, nBand);
MI_aperiodic = nan(size(periodic_ga,1), nCondFull, nBand);
MI_total     = nan(size(periodic_ga,1), nCondFull, nBand);

p_periodic  = nan(nBand, nCondDiff); sig_periodic  = false(nBand, nCondDiff); eff_periodic  = nan(nBand, nCondDiff);
p_aperiodic = nan(nBand, nCondDiff); sig_aperiodic = false(nBand, nCondDiff); eff_aperiodic = nan(nBand, nCondDiff);
p_total     = nan(nBand, nCondDiff); sig_total     = false(nBand, nCondDiff); eff_total     = nan(nBand, nCondDiff);
p_direct    = nan(nBand, nCondDiff); sig_direct    = false(nBand, nCondDiff); eff_direct    = nan(nBand, nCondDiff);

figure('Color','w','Position',[100 -400 380 260*nBand]);

for b = 1:nBand
    bl_per = periodic_ga(:,1,b);
    bl_ape = aperiodic_ga(:,1,b);
    bl_tot = total_ga(:,1,b);

    MI_periodic(:,1,b)  = 0;   % baseline vs itself = 0 by definition
    MI_aperiodic(:,1,b) = 0;
    MI_total(:,1,b)     = 0;

    MI_periodic(:,2:end,b)  = (periodic_ga(:,2:end,b)  - bl_per) ./ (abs(periodic_ga(:,2:end,b))  + abs(bl_per));
    MI_aperiodic(:,2:end,b) = (aperiodic_ga(:,2:end,b) - bl_ape) ./ (abs(aperiodic_ga(:,2:end,b)) + abs(bl_ape));
    MI_total(:,2:end,b)     = (total_ga(:,2:end,b)     - bl_tot) ./ (abs(total_ga(:,2:end,b))     + abs(bl_tot));

    % --- Test 1a/1b/1c: each measure's own MI vs zero, per condition (R1-R4)
    [p_periodic(b,:), sig_periodic(b,:), eff_periodic(b,:)]   = test_vs_zero(MI_periodic(:,2:end,b));
    [p_aperiodic(b,:), sig_aperiodic(b,:), eff_aperiodic(b,:)] = test_vs_zero(MI_aperiodic(:,2:end,b));
    [p_total(b,:), sig_total(b,:), eff_total(b,:)]             = test_vs_zero(MI_total(:,2:end,b));

    % --- Test 2: DIRECT periodic vs aperiodic comparison, per condition
    p_raw = nan(1, nCondDiff);
    eff_raw = nan(1, nCondDiff);
    for c = 1:nCondDiff
        pair_diff = MI_periodic(:,c+1,b) - MI_aperiodic(:,c+1,b);
        pair_diff = pair_diff(~isnan(pair_diff) & isfinite(pair_diff));
        if numel(pair_diff) >= 3
            p_raw(c) = signrank(pair_diff);
            eff_raw(c) = rank_biserial(pair_diff);
        end
    end
    [~, sig_c] = local_fdr(p_raw);
    p_direct(b,:)   = p_raw;
    sig_direct(b,:) = sig_c;
    eff_direct(b,:) = eff_raw;

    % --- plot: 3 overlaid lines, mean +/- SEM ribbons
    subplot(nBand, 1, b);
    hold on;
    x = 1:nCondFull;

    plot_data = {MI_periodic(:,:,b), 'Periodic'; MI_aperiodic(:,:,b), 'Aperiodic'; MI_total(:,:,b), 'Total'};
    h = gobjects(3,1);
    mean_stack = nan(3, nCondFull);
    all_extent = [];
    for m2 = 1:3
        d = plot_data{m2,1};
        mean_d = mean(d, 1, 'omitnan');
        n_valid = sum(~isnan(d), 1);
        sem_d = std(d, 0, 1, 'omitnan') ./ sqrt(max(n_valid,1));
        fill([x, fliplr(x)], [mean_d+sem_d, fliplr(mean_d-sem_d)], measure_colors(m2,:), ...
            'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        h(m2) = plot(x, mean_d, '-', 'Color', measure_colors(m2,:), 'LineWidth', 1.8, 'MarkerFaceColor', measure_colors(m2,:));
        mean_stack(m2,:) = mean_d;
        all_extent = [all_extent, mean_d+sem_d, mean_d-sem_d]; %#ok<AGROW>
    end
    yline(0, 'k--', 'HandleVisibility', 'off');

    % --- symmetric, data-driven y-axis based on all 3 measures' extent in this band
    M = max(abs(all_extent));
    yrange = [-M, M] * 1.1;

    yrange = [-0.6 0.6];
    ylim(yrange);
    yt = linspace(min(yrange), max(yrange), 5);
    yticks(yt);

    % --- small color-matched markers: measure's own MI significantly != 0
    own_sig = {sig_periodic(b,:), 1; sig_aperiodic(b,:), 2; sig_total(b,:), 3};
    for m2 = 1:3
        sig_this = own_sig{m2,1};
        for c = 1:nCondDiff
            if sig_this(c)
                plot(c+1, mean_stack(m2,c+1), 'o', 'MarkerSize', 8, 'MarkerEdgeColor', measure_colors(m2,:), ...
                    'MarkerFaceColor', measure_colors(m2,:), 'HandleVisibility', 'off');
            end
        end
    end

    % --- prominent black star: DIRECT periodic-vs-aperiodic comparison
    for c = 1:nCondDiff
        if sig_c(c)
            y_top = max(mean_stack(1,c+1), mean_stack(2,c+1));
            text(c+1, y_top + 0.08*range(ylim), '*', 'Color', 'k', 'FontSize', 16, ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
    end

    set(gca, 'XTick', 1:nCondFull, 'XTickLabel', cond_labels_full);
    title(bands(b).name);
    ylabel('Modulation index vs baseline');
    box off;
    if b == 1, legend(h, plot_data(:,2), 'Location', 'best'); end
end

sgtitle(titletext, 'FontSize', 13, 'FontWeight', 'bold');
annotation('textbox', [0.13 0.005 0.8 0.04], 'String', ...
    'filled dot (color-matched) = that measure differs from its own baseline  |  black * = periodic differs from aperiodic directly', ...
    'FontSize', 8, 'EdgeColor', 'none');

% -------------------------------------------------------------------------
% STATS REPORT TEXT
% -------------------------------------------------------------------------
cond_labels_diff = cond_labels_full(2:end);
report_txt = '';
report_txt = [report_txt, sprintf('Title: %s\n', titletext)];
report_txt = [report_txt, sprintf('Generated: %s\n\n', datestr(now))];
report_txt = [report_txt, sprintf('Effect size: matched-pairs rank-biserial correlation (Kerby, 2014), range [-1, 1]\n\n')];

for b = 1:nBand
    report_txt = [report_txt, sprintf('=== %s ===\n', bands(b).name)];
    report_txt = [report_txt, sprintf('%-8s %-30s %-30s %-30s %-30s\n', 'Cond', 'Periodic vs baseline', 'Aperiodic vs baseline', 'Total vs baseline', 'Periodic vs Aperiodic (direct)')];
    report_txt = [report_txt, sprintf('%s\n', repmat('-', 1, 130))];
    for c = 1:nCondDiff
        report_txt = [report_txt, sprintf('%-8s %-30s %-30s %-30s %-30s\n', cond_labels_diff{c}, ...
            fmt(p_periodic(b,c),  sig_periodic(b,c),  eff_periodic(b,c)), ...
            fmt(p_aperiodic(b,c), sig_aperiodic(b,c), eff_aperiodic(b,c)), ...
            fmt(p_total(b,c),     sig_total(b,c),     eff_total(b,c)), ...
            fmt(p_direct(b,c),    sig_direct(b,c),    eff_direct(b,c)))];
    end
    report_txt = [report_txt, sprintf('\n')];
end

% -------------------------------------------------------------------------
% OUTPUTS
% -------------------------------------------------------------------------
stats_own     = struct();
stats_own.p   = struct('periodic', p_periodic,   'aperiodic', p_aperiodic,   'total', p_total);
stats_own.sig = struct('periodic', sig_periodic, 'aperiodic', sig_aperiodic, 'total', sig_total);
stats_own.eff = struct('periodic', eff_periodic, 'aperiodic', eff_aperiodic, 'total', eff_total);

stats_direct = struct('p', p_direct, 'sig', sig_direct, 'eff', eff_direct);

MI_all = struct('periodic', MI_periodic, 'aperiodic', MI_aperiodic, 'total', MI_total);

end

% -------------------------------------------------------------------
function [p_raw, sig, eff] = test_vs_zero(d)
% one-sample Wilcoxon signed-rank per column (condition), FDR-gated,
% with rank-biserial correlation effect size (Kerby, 2014)
nCond = size(d,2);
p_raw = nan(1, nCond);
eff   = nan(1, nCond);
for c = 1:nCond
    col = d(:,c); col = col(~isnan(col) & isfinite(col));
    if numel(col) >= 3
        p_raw(c) = signrank(col);
        eff(c)   = rank_biserial(col);
    end
end
[~, sig] = local_fdr(p_raw);
end

function r = rank_biserial(col)
% matched-pairs rank-biserial correlation: r = (W+ - W-) / (W+ + W-)
nz = col(col ~= 0);
r = nan;
if ~isempty(nz)
    ranks = tiedrank(abs(nz));
    w_pos = sum(ranks(nz > 0));
    w_neg = sum(ranks(nz < 0));
    r = (w_pos - w_neg) / (w_pos + w_neg);
end
end

function [crit_p, sig] = local_fdr(p)
valid = ~isnan(p);
p_sorted = sort(p(valid));
m = numel(p_sorted);
if m == 0
    crit_p = 0; sig = false(size(p));
    return
end
below = p_sorted <= (1:m)/m * 0.05;
if any(below), crit_p = p_sorted(find(below,1,'last')); else, crit_p = 0; end
sig = valid & (p <= crit_p);
end

function s = fmt(p, sig, r)
if isnan(p)
    s = 'n/a';
else
    sig_str = ''; if sig, sig_str = '*(FDR)'; end
    s = sprintf('p=%.4f %s r=%.3f', p, sig_str, r);
end
end