function [report_txt, p_vs_baseline, sig_vs_baseline, effect_vs_baseline, MI] = plot_MI_single(measure_ga, cond_labels_full, titletext)
% PLOT_MI_SINGLE  Modulation index MI = (condition - baseline) / (|condition|
% + |baseline|) for a single BANDLESS measure (e.g. exp_ga, nSubj x
% nCondFull, no band dimension). Only tests baseline-vs-R1...R4 (no
% direct comparison between two measures, since there's only one here).
%
% Absolute values in the denominator guarantee MI stays within [-1, 1]
% regardless of sign (triangle inequality) -- same reasoning as the
% multi-measure version.
%
% Usage:
%   [p_vs_baseline, sig_vs_baseline, effect_vs_baseline, MI, report_txt] = ...
%       plot_MI_single(exp_ga, cond_labels_full, 'Exponent modulation index');
%
% Inputs:
%   measure_ga       - nSubj x nCondFull, RAW units, baseline in column 1
%   cond_labels_full - {'baseline','R1','R2','R3','R4'}
%   titletext        - figure title
%
% Outputs:
%   p_vs_baseline, sig_vs_baseline, effect_vs_baseline - 1 x nCondDiff
%       (one-sample Wilcoxon signed-rank test per condition, FDR-gated
%       across the nCondDiff comparisons; effect = rank-biserial
%       correlation, Kerby 2014)
%   MI         - nSubj x nCondFull (baseline column is exactly 0)
%   report_txt - single multi-line string, ready to write to a text file

nCondFull = size(cond_labels_full, 2);
nCondDiff = nCondFull - 1;

bl = measure_ga(:,1);

MI = nan(size(measure_ga));
MI(:,1) = 0;   % baseline vs itself = 0 by definition
MI(:,2:end) = (measure_ga(:,2:end) - bl) ./ (abs(measure_ga(:,2:end)) + abs(bl));

% --- test: MI vs zero, per condition, FDR-gated across the nCondDiff comparisons
p_raw = nan(1, nCondDiff);
eff_raw = nan(1, nCondDiff);
for c = 1:nCondDiff
    col = MI(:,c+1); col = col(~isnan(col) & isfinite(col));
    if numel(col) >= 3
        p_raw(c) = signrank(col);
        eff_raw(c) = rank_biserial(col);
    end
end
[~, sig_c] = local_fdr(p_raw);
p_vs_baseline      = p_raw;
sig_vs_baseline    = sig_c;
effect_vs_baseline = eff_raw;

% ============================================================
%  REPORTING
% ============================================================
lines = {};
lines{end+1} = sprintf('=== %s: MI vs baseline, %d comparisons, FDR-corrected ===', titletext, nCondDiff);
lines{end+1} = sprintf('%-10s %-12s %-10s %-12s', 'Cond', 'p-value', 'sig(FDR)', 'effect_r');
lines{end+1} = repmat('-', 1, 46);
for c = 1:nCondDiff
    if isnan(p_raw(c))
        lines{end+1} = sprintf('%-10s %-12s %-10s %-12s', cond_labels_full{c+1}, 'n/a', 'n/a', 'n/a'); %#ok<AGROW>
    else
        sig_str = ''; if sig_c(c), sig_str = 'yes'; else, sig_str = 'no'; end
        lines{end+1} = sprintf('%-10s %-12.4f %-10s %-12.3f', cond_labels_full{c+1}, p_raw(c), sig_str, eff_raw(c)); %#ok<AGROW>
    end
end

report_txt = sprintf('%s\n', lines{:});
fprintf('\n%s\n', report_txt);

% ============================================================
%  PLOT: single line + shaded ribbon, symmetric y-axis
% ============================================================
figure('Color','w','Position',[100 100 450 380]);
hold on;

x = 1:nCondFull;
mean_d  = mean(MI, 1, 'omitnan');
n_valid = sum(~isnan(MI), 1);
sem_d   = std(MI, 0, 1, 'omitnan') ./ sqrt(max(n_valid,1));

fill([x, fliplr(x)], [mean_d+sem_d, fliplr(mean_d-sem_d)], [0.2 0.4 0.7], ...
    'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(x, mean_d, '-o', 'Color', [0.2 0.4 0.7], 'LineWidth', 1.8, 'MarkerFaceColor', [0.2 0.4 0.7]);
yline(0, 'k--', 'HandleVisibility', 'off');

% --- symmetric, data-driven y-axis
M = max(abs([mean_d+sem_d, mean_d-sem_d]));
yrange = [-M, M] * 1.1;

for i = 1:2
    yrange(i) = round(yrange(i),2);
end

ylim(yrange);
yt = linspace(min(yrange), max(yrange), 5);
yticks(yt);

% --- significance markers
for c = 1:nCondDiff
    if sig_c(c)
        text(c+1, mean_d(c+1) + sem_d(c+1) + 0.08*range(ylim), '*', 'Color', 'k', ...
            'FontSize', 16, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
end

set(gca, 'XTick', 1:nCondFull, 'XTickLabel', cond_labels_full);
ylabel('Modulation index vs baseline');
title(titletext, 'Interpreter', 'none');
box off; grid on;

end

% -------------------------------------------------------------------
function r = rank_biserial(col)
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
p_for_fdr = p;
p_for_fdr(~valid) = 1;
p_sorted = sort(p_for_fdr);
m = numel(p_sorted);
below = p_sorted <= (1:m)/m * 0.05;
if any(below), crit_p = p_sorted(find(below,1,'last')); else, crit_p = 0; end
sig = valid & (p_for_fdr <= crit_p);
end
