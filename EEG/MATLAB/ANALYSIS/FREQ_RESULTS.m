%% SETUP

clear all; clc; close all;

restoredefaultpath;
addpath 'PATH_TO_FIELDTRIP';
ft_defaults;

load('reset_elec_63.mat');

FREQ_OUT    = '../FREQ_OUT/';

freq_files = dir(fullfile(FREQ_OUT, '*_freq.mat'));
fprintf('Found %d freq files\n', numel(freq_files));

RESULTS_DIR = fullfile(FREQ_OUT, 'results_nov_10');
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

%% ---------------- LOAD & MERGE ----------------

R = [];
for i = 1:numel(freq_files)
    tmp = load(fullfile(freq_files(i).folder, freq_files(i).name));
    if isempty(R), R = tmp.R; else, R(end+1) = tmp.R; end 
end
fprintf('Loaded %d sessions\n', numel(R));

pid_nr_map = containers.Map();
pidNr = 0;
for i = 1:numel(R)
    pid_short = R(i).PID(1:6);
    if ~isKey(pid_nr_map, pid_short)
        pidNr = pidNr + 1;
        pid_nr_map(pid_short) = pidNr;
    end
    R(i).PID_nr = pid_nr_map(pid_short);
end
fprintf('Unique subjects: %d\n', pidNr);

% % select bw sessions
R_sel = R([R.sesh_bw_true]);
nSubj = numel(R_sel);

% select pmr sessions
%R_sel = R(~[R.sesh_bw_true]);
%nSubj = numel(R_sel);

%% ---------------- ANALYSIS SCOPE: 5 rated conditions, 5 bands ----------------
conds       = 2:6;
nCond_plt   = numel(conds);
cond_labels = {'B','R1','R2','R3','R4'};

bands = struct();
bands(1).name = 'delta'; bands(1).range = [1 4];
bands(2).name = 'theta'; bands(2).range = [4 8];
bands(3).name = 'alpha'; bands(3).range = [8 13];
bands(4).name = 'beta';  bands(4).range = [13 30];
bands(5).name = 'gamma'; bands(5).range = [30 45];
nBand = numel(bands);

% color map- lana
% +++++++ COLORMAP +++++++
cmap_seq = parula(30);
cmap_band = cmap_seq(18:3:30,:);
cmap_cond = cmap_seq(4:9,:);

cmap_cond = flipud(cmap_cond);

cmap_band = parula(100); cmap_band = cmap_band(30:15:100,:);
cmap_cond = parula(100); cmap_cond = cmap_cond(10:15:80,:);
clr_gray = [0.2 0.2 0.2];

figure;
for i = 1:5
subplot(2,1,1);
plot(1:100,ones(1,100)*i, 'Color', cmap_band(i,:) ,'LineWidth',3); ylim([0 6]); hold on;
subplot(2,1,2);
plot(1:100,ones(1,100)*i, 'Color', cmap_cond(i,:) ,'LineWidth',3); ylim([0 6]); hold on;
end

%  ASC RATINGS 
rated_idx = 2:6;
nCondAll  = 7;

asc = struct();
asc.start = nan(nSubj,nCondAll);
asc.end   = nan(nSubj,nCondAll);
asc.mean   = nan(nSubj,nCondAll);
asc.mean_z   = nan(nSubj,nCondAll);
for i = 1:nSubj
    pairs = reshape(R_sel(i).asc_ratings, 2, numel(rated_idx))';
    asc.start(i,rated_idx) = pairs(:,1)';
    asc.end(i,rated_idx)   = pairs(:,2)';
    
    asc.mean(i,rated_idx)   = mean(pairs(:,1:2)');
    asc.mean_z(i,rated_idx) = zscore(asc.mean(i,rated_idx));
end

asc.delta_wthn = asc.end - asc.start;
asc.delta_btwn = diff(asc.mean,1,2);

asc.delta_wthn_cat = asc.delta_wthn;
asc.delta_wthn_cat(asc.delta_wthn<0) = -1;
asc.delta_wthn_cat(asc.delta_wthn>0) = 1;

asc.delta_btwn_cat = asc.delta_btwn;
asc.delta_btwn_cat(asc.delta_btwn<0) = -1;
asc.delta_btwn_cat(asc.delta_btwn>0) = 1;


% --- tertile categories on ASC mean (pooled across all subjects*rated-trials)
asc_sub    = asc.mean(:, rated_idx);              % nSubj x 5, aligned with measure_ga's columns
tert_edges = quantile(asc_sub(:), [1/3, 2/3]);     % pooled tertile cutpoints, not per-subject

tert_tmp = nan(size(asc_sub));
tert_tmp(asc_sub <= tert_edges(1))                              = 1;   % low
tert_tmp(asc_sub > tert_edges(1) & asc_sub <= tert_edges(2))    = 2;   % mid
tert_tmp(asc_sub > tert_edges(2))                               = 3;   % high

asc.mean_tertile = nan(size(asc.mean));
asc.mean_tertile(:, rated_idx) = tert_tmp;

lohi_tmp = nan(size(asc_sub));
lohi_tmp(asc_sub <= 2.5)                              = 1;   % low
lohi_tmp(asc_sub > 2.5)                               = 2;   % high

asc.mean_tertile = nan(size(asc.mean));
asc.mean_tertile(:, rated_idx) = tert_tmp;

asc.mean_lohi = nan(size(asc.mean));
asc.mean_lohi(:, rated_idx) = lohi_tmp;

fprintf('ASC mean tertile cutpoints: low <= %.2f, mid <= %.2f, high above\n', tert_edges(1), tert_edges(2));

%% ---------------- QC: r2 / epoch counts (7 conditions, own scope) ----------------
cond_labels_qc = {'EOpre','ECpre','R1','R2','R3','ECpost','EOpost'};
r2_mat       = nan(nSubj, 7);
n_epochs_all = nan(nSubj, 7);
for s = 1:nSubj
    for c = 1:7
        r2_mat(s,c)       = nanmean(R_sel(s).fooof_r2{c}(:));
        n_epochs_all(s,c) = R_sel(s).n_epochs_used(c);
    end
end
fprintf('Mean r2 overall: %.4f\n', mean(r2_mat(:), 'omitnan'));


% -- anova on number of epochs on which analyses were run >> no main effect
% of condition 
t_qc = array2table(n_epochs_all, 'VariableNames', cond_labels_qc);
within_qc = table(cond_labels_qc', 'VariableNames', {'Condition'});
rm_qc = fitrm(t_qc, [cond_labels_qc{1} '-' cond_labels_qc{end} ' ~ 1'], 'WithinDesign', within_qc);
disp(ranova(rm_qc, 'WithinModel', 'Condition'));

figure; boxplot(n_epochs_all)

%% ---------------- BUILD TOPO + GA MATRICES: raw / aperiodic / foofed AUC, + exponent ----------------


broad_total_auc_topos = nan(63, nSubj, nCond_plt);
broad_fooof_auc_topos = nan(63, nSubj, nCond_plt);
broad_aper_auc_topos = nan(63, nSubj, nCond_plt);


exp_topos       = nan(63, nSubj, nCond_plt);
offset_topos    = nan(63, nSubj, nCond_plt);
total_auc_topos = nan(63, nSubj, nCond_plt, nBand);
aper_auc_topos  = nan(63, nSubj, nCond_plt, nBand);
fooof_auc_topos = nan(63, nSubj, nCond_plt, nBand);
 
for ci = 1:nCond_plt
    c = conds(ci);
    for s = 1:nSubj

%         broad_raw_topos(:,s,ci) = mean(R_sel(s).broad_pwr{c}, 2, 'omitnan');
        
        broad_total_auc_topos(:,s,ci) = mean(R_sel(s).broad_total_auc{c}, 2, 'omitnan');
        broad_fooof_auc_topos(:,s,ci) = mean(R_sel(s).broad_fooof_auc{c}, 2, 'omitnan');
        broad_aper_auc_topos(:,s,ci) = mean(R_sel(s).broad_aper_auc{c}, 2, 'omitnan');
        
        exp_topos(:,s,ci) = R_sel(s).exponent{c};
        offset_topos(:,s,ci) = R_sel(s).offset{c};
        
        
        for b = 1:nBand
            bn = bands(b).name;
            total_auc_topos(:,s,ci,b) = R_sel(s).band_total_auc.(bn){c};
            aper_auc_topos(:,s,ci,b)  = R_sel(s).band_aper_auc.(bn){c};
            fooof_auc_topos(:,s,ci,b) = R_sel(s).band_fooof_auc.(bn){c};
        end
    end
end
 
broad_total_auc_ga    = squeeze(mean(broad_total_auc_topos, 1, 'omitnan'));  
broad_fooof_auc_ga    = squeeze(mean(broad_fooof_auc_topos, 1, 'omitnan'));  
broad_aper_auc_ga    = squeeze(mean(broad_aper_auc_topos, 1, 'omitnan'));  

exp_ga       = squeeze(mean(exp_topos, 1, 'omitnan'));        % nSubj x nCond
offset_ga    = squeeze(mean(offset_topos, 1, 'omitnan'));        % nSubj x nCond
total_auc_ga = squeeze(mean(total_auc_topos, 1, 'omitnan'));  % nSubj x nCond x nBand
aper_auc_ga  = squeeze(mean(aper_auc_topos, 1, 'omitnan'));
fooof_auc_ga = squeeze(mean(fooof_auc_topos, 1, 'omitnan'));
 
valid_all = squeeze(~any(any(isnan(exp_topos), 1), 3));
fprintf('Subjects valid for cluster tests: %d / %d\n', sum(valid_all), nSubj);

%% R2 dependency

clc;
% does aperiodic power correlate with fit quality, per band?
for b = 1:nBand
    bn = bands(b).name;
    
    total_vals = arrayfun(@(s) nanmean(cell2mat(cellfun(@(x) mean(x,'omitnan'), R_sel(s).band_total_auc.(bn), 'UniformOutput', false))), 1:nSubj)';
    per_vals = arrayfun(@(s) nanmean(cell2mat(cellfun(@(x) mean(x,'omitnan'), R_sel(s).band_fooof_auc.(bn), 'UniformOutput', false))), 1:nSubj)';
    aper_vals = arrayfun(@(s) nanmean(cell2mat(cellfun(@(x) mean(x,'omitnan'), R_sel(s).band_aper_auc.(bn), 'UniformOutput', false))), 1:nSubj)';
    
    r2_vals   = arrayfun(@(s) nanmean(cell2mat(cellfun(@(x) mean(x,'omitnan'), R_sel(s).fooof_r2, 'UniformOutput', false))), 1:nSubj)';

    [r, p] = corr(total_vals, r2_vals, 'rows', 'complete');
    fprintf('%s: total power vs r2, r=%.3f, p=%.4f\n', bn, r, p);

    [r, p] = corr(aper_vals, r2_vals, 'rows', 'complete');
    fprintf('%s: aperiodic power vs r2, r=%.3f, p=%.4f\n', bn, r, p);
    
    [r, p] = corr(per_vals, r2_vals, 'rows', 'complete');
    fprintf('%s: periodic power vs r2, r=%.3f, p=%.4f\n', bn, r, p);
end


for c = 1:5   % or in selectrd conditions 
    r2_this_cond = arrayfun(@(s) nanmean(R_sel(s).fooof_r2{conds(c)}), 1:nSubj);
    fprintf('%s: mean r2 = %.4f\n', cond_labels{c}, nanmean(r2_this_cond));
end
%% FIG. 2A) FULL-SPECTRUM PSD PANEL (raw / aperiodic-line / foofed-residual) ----------------

% Uses aper_curve_lin directly -- no 10.^ conversion needed anymore.

close all;

n = nSubj;
freq_vec = R_sel(1).freq_vec;
nFreq    = numel(freq_vec);

psd_chavg  = nan(n, nFreq, nCond_plt);
aper_chavg = nan(n, nFreq, nCond_plt);
osc_chavg  = nan(n, nFreq, nCond_plt);

for i = 1:n
    for ci = 1:nCond_plt
        c = conds(ci);
        psd_chavg(i,:,ci)  = mean(R_sel(i).broad_pwr{c}, 1, 'omitnan');      % PSD, mean across epochs
        aper_chavg(i,:,ci) = mean(R_sel(i).aper_curve_lin{c}, 1, 'omitnan'); % aperiodic curve, linear(
        osc_chavg(i,:,ci)  = mean(R_sel(i).osc_pwr{c}, 1, 'omitnan');        % residual
    end
end

psd_mean = squeeze(mean(psd_chavg, 1, 'omitnan'))';
psd_sem  = (squeeze(std(psd_chavg, 0, 1, 'omitnan')) ./ sqrt(n))';
aper_mean = squeeze(mean(aper_chavg, 1, 'omitnan'))';
osc_mean  = squeeze(mean(osc_chavg, 1, 'omitnan'))';
osc_sem   = (squeeze(std(osc_chavg, 0, 1, 'omitnan')) ./ sqrt(n))';


fig = figure('Color', 'w', 'Position', [100 100 900 800]);
t = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
band_colors = cmap_band;

% --- shared y-range for panels 1 & 2 (raw PSD + fitted aperiodic line),
% computed from the raw data BEFORE plotting anything -- can't rely on
% ylim() auto-scaling afterward, since aper_mean is never plotted in
% ax_big, only in ax_fit, so ax_big's auto-range wouldn't reflect it.
yl_shared = [min([psd_mean(:)-psd_sem(:); aper_mean(:)], [], 'omitnan'), ...
             max([psd_mean(:)+psd_sem(:); aper_mean(:)], [], 'omitnan')];

% Greek symbol lookup for band labels, instead of full text names
band_symbols = struct('delta','\delta','theta','\theta','alpha','\alpha','beta','\beta','gamma','\gamma');

ax_big = nexttile(t);
hold(ax_big, 'on');
h = gobjects(nCond_plt,1);
for ci = 1:nCond_plt
    fill(ax_big, [freq_vec, fliplr(freq_vec)], [psd_mean(ci,:)+psd_sem(ci,:), fliplr(psd_mean(ci,:)-psd_sem(ci,:))], ...
        cmap_cond(ci,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    h(ci) = plot(ax_big, freq_vec, psd_mean(ci,:), 'Color', cmap_cond(ci,:), 'LineWidth', 1.8);
end
set(ax_big, 'YScale', 'log');
xlabel(ax_big, 'Frequency (Hz)'); ylabel(ax_big, 'Power (\muV^2)');
legend(ax_big, h, cond_labels, 'Location', 'northeast','autoupdate','off');
box(ax_big,'off'); grid(ax_big,'on');
for b = 1:numel(bands)
    r = bands(b).range;
    text(ax_big, sqrt(r(1)*r(2)), yl_shared(2), band_symbols.(bands(b).name), 'HorizontalAlignment','center', ...
        'VerticalAlignment','top', 'FontSize', 12, 'Color', [0.3 0.3 0.3]);
    xline(r(1),'--k');
end
set(ax_big, 'FontSize',12, 'TickLength', [0.03 0.03], 'XLim', [0 45], 'XTick',[0 15 30 45], 'YLim', yl_shared);
axis(ax_big, 'square');

% ======== EXPONENT ========
ax_fit = nexttile(t);
hold(ax_fit, 'on');
for ci = 1:nCond_plt
    plot(ax_fit, freq_vec, aper_mean(ci,:), 'Color', cmap_cond(ci,:), 'LineWidth', 1.5);
end
for b = 1:numel(bands)
    r = bands(b).range;
    xline(r(1),'--k');
end
set(ax_fit, 'YScale', 'log');
xlabel(ax_fit, 'Frequency (Hz)'); ylabel(ax_fit, 'Power (\muV^2)');
box(ax_fit,'off'); grid(ax_fit,'on');
set(ax_fit, 'FontSize',12, 'TickLength', [0.03 0.03], 'XLim', [0 45], 'XTick',[0 15 30 45], 'YLim', yl_shared);
axis(ax_fit, 'square');

% ======== RESIDUAL (OSC) ==========
ax_osc = nexttile(t);
hold(ax_osc, 'on');
for ci = 1:nCond_plt
    fill(ax_osc, [freq_vec, fliplr(freq_vec)], [osc_mean(ci,:)+osc_sem(ci,:), fliplr(osc_mean(ci,:)-osc_sem(ci,:))], ...
        cmap_cond(ci,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(ax_osc, freq_vec, osc_mean(ci,:), 'Color', cmap_cond(ci,:), 'LineWidth', 1.5);
end
for b = 1:numel(bands)
    r = bands(b).range;
    xline(r(1),'--k');
end
yline(ax_osc, 0, 'k:');
xlabel(ax_osc, 'Frequency (Hz)'); ylabel(ax_osc, 'log_{10}(raw / aperiodic)');
box(ax_osc,'off'); grid(ax_osc,'on');
set(ax_osc, 'FontSize',12, 'TickLength', [0.03 0.03], 'XLim', [0 45], 'XTick',[0 15 30 45]);
axis(ax_osc, 'square');   % NOT y-linked to the other two -- different units (log-ratio, not power), so left on its own auto-scaled range

linkaxes([ax_big ax_fit ax_osc], 'x');
saveas(gcf, fullfile(RESULTS_DIR, ['PSD-ALL' '.svg']),'svg');
% close all;

%% FIG. 2B) GATHERED PLOTS: TOTAL (RAW PSD AUC) - PERIO (FOOFED AUC) - APERIO (EXPONENT AUC)


% --- exponent ----
close all; 
analysis_id = 'MI_exponent';
report_txt = [];
 
report_txt = plot_MI_single(exp_ga, cond_labels, 'Exponent modulation index');

saveas(gcf, fullfile(RESULTS_DIR, [analysis_id '.svg']),'svg'); close(gcf);

fid = fopen(fullfile(RESULTS_DIR, [analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt);
fclose(fid);

% --- PERIO APERIO TOTAL BROADBAND

close all; 
analysis_id = 'MI_broadband_per_aper_total';
 
bdummy = struct(); bdummy.name = 'broadband'; % bcs function expects bands

report_txt = [];
report_txt = plot_MI_comparison( ...
    broad_fooof_auc_ga, broad_aper_auc_ga, broad_total_auc_ga, cond_labels, bdummy, 'Modulation index vs baseline');

saveas(gcf, fullfile(RESULTS_DIR, [analysis_id '.svg']),'svg'); close(gcf);
fid = fopen(fullfile(RESULTS_DIR, [analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt);
fclose(fid);

% ------- PERIO APERIO TOTAL PER BAND
close all; 
analysis_id = 'MI_periodic_aperiodic_total';

report_txt = [];
report_txt = plot_MI_comparison( ...
    fooof_auc_ga, aper_auc_ga, total_auc_ga, cond_labels, bands, 'Modulation index vs baseline');

saveas(gcf, fullfile(RESULTS_DIR, [analysis_id '.svg']),'svg'); close(gcf);

fid = fopen(fullfile(RESULTS_DIR, [analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt);
fclose(fid);
%% FIG. 2C) ALL PWR TOPOS

close all;

% ===== broadband =====

% --- total ---
analysis_id = 'broadband_total';
topos_this = broad_total_auc_topos;
report_txt = [];
% >> topos
report_txt = fun_plot_topos_T_stats(topos_this, cond_labels, chan_labels_63, layout_63, neighbours_63, analysis_id);
% >> figure
saveas(gcf, fullfile(RESULTS_DIR, ['topoT_' analysis_id '.svg']),'svg'); close(gcf);
% >> text
fid = fopen(fullfile(RESULTS_DIR, ['clusterstats_' analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt); fclose(fid);

% --- period ---
analysis_id = 'broadband_perio';
topos_this = broad_fooof_auc_topos;
report_txt = [];
% >> topos
report_txt = fun_plot_topos_T_stats(topos_this, cond_labels, chan_labels_63, layout_63, neighbours_63, analysis_id);
% >> figure
saveas(gcf, fullfile(RESULTS_DIR, ['topoT_' analysis_id '.svg']),'svg'); close(gcf);
% >> text
fid = fopen(fullfile(RESULTS_DIR, ['clusterstats_' analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt); fclose(fid);

% --- aperio ---
analysis_id = 'broadband_aperio';
topos_this = broad_aper_auc_topos;
report_txt = [];
% >> topos
report_txt = fun_plot_topos_T_stats(topos_this, cond_labels, chan_labels_63, layout_63, neighbours_63, analysis_id);
% >> figure
saveas(gcf, fullfile(RESULTS_DIR, ['topoT_' analysis_id '.svg']),'svg'); close(gcf);
% >> text
fid = fopen(fullfile(RESULTS_DIR, ['clusterstats_' analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt); fclose(fid);

% ===== exponent =====
analysis_id = 'exponent';
topos_this = exp_topos;
report_txt = [];
% >> topos
report_txt = fun_plot_topos_T_stats(topos_this, cond_labels, chan_labels_63, layout_63, neighbours_63, analysis_id);
% >> figure
saveas(gcf, fullfile(RESULTS_DIR, ['topoT_' analysis_id '.svg']),'svg'); close(gcf);
% >> text
fid = fopen(fullfile(RESULTS_DIR, ['clusterstats_' analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt); fclose(fid);

% ++++++ PER BAND +++++++
zlims = [];
for b = 1:nBand
    % ===== total_auc =====
    analysis_id = ['total_auc_' bands(b).name];
    topos_this = total_auc_topos(:, :, :, b);
    report_txt = [];
    % >> topos
    report_txt = fun_plot_topos_T_stats(topos_this, cond_labels, chan_labels_63, layout_63, neighbours_63, analysis_id);
    % >> figure
    saveas(gcf, fullfile(RESULTS_DIR, ['topoT_' analysis_id '.svg']),'svg'); close(gcf);
    % >> text
    fid = fopen(fullfile(RESULTS_DIR, ['clusterstats_' analysis_id '.txt']), 'w');
    fprintf(fid, '%s', report_txt); fclose(fid);   
end

% disp(max(max(zlims))) % 5

zlims = [];
for b = 1:nBand    
    % ===== aperio_auc =====
    analysis_id = ['aperio_auc' bands(b).name];
    topos_this = aper_auc_topos(:, :, :, b);
    report_txt = [];
    % >> topos
    report_txt = fun_plot_topos_T_stats(topos_this, cond_labels, chan_labels_63, layout_63, neighbours_63, analysis_id);
    % >> figure
    saveas(gcf, fullfile(RESULTS_DIR, ['topoT_' analysis_id '.svg']),'svg'); close(gcf);
    % >> text
    fid = fopen(fullfile(RESULTS_DIR, ['clusterstats_' analysis_id '.txt']), 'w');
    fprintf(fid, '%s', report_txt); fclose(fid);
end

% disp(max(max(zlims))) % 4.2

zlims = [];
for b = 1:nBand
    % ===== fooof_auc =====
    analysis_id = ['fooof_auc' bands(b).name];
    topos_this = fooof_auc_topos(:, :, :, b);
    report_txt = [];
    % >> topos
    report_txt = fun_plot_topos_T_stats(topos_this, cond_labels, chan_labels_63, layout_63, neighbours_63, analysis_id);
    % >> figure
    saveas(gcf, fullfile(RESULTS_DIR, ['topoT_' analysis_id '.svg']),'svg'); close(gcf);
    % >> text
    fid = fopen(fullfile(RESULTS_DIR, ['clusterstats_' analysis_id '.txt']), 'w');
    fprintf(fid, '%s', report_txt); fclose(fid);

end
% disp(max(max(zlims))) 

%% + ASC TOPO relationships (x NOT USED x)

close all; clc;

asc_meas_iv = asc.start(:,rated_idx);

% ===== exponent =====
analysis_id = 'ASCstart_exponent_r';
topos_this = exp_topos;
report_txt = [];

% >> topos (all bands)
report_txt = fun_plot_topos_iv_regression(topos_this, asc_meas_iv, chan_labels_63, layout_63, neighbours_63, 'exponent',analysis_id);
% >> figure
saveas(gcf, fullfile(RESULTS_DIR, ['regression_' analysis_id '.svg']),'svg'); close(gcf);
% >> text
fid = fopen(fullfile(RESULTS_DIR, ['regression_' analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt); fclose(fid);


% ===== total_auc =====
analysis_id = 'ASCstart_total_auc_r';
topos_band = total_auc_topos;
report_txt = [];

% >> topos (all bands)
report_txt = fun_plot_topos_iv_regression_bands(topos_band, asc_meas_iv, chan_labels_63, layout_63, neighbours_63, {bands.name}, analysis_id);
% >> figure
saveas(gcf, fullfile(RESULTS_DIR, ['regression_' analysis_id '.svg']),'svg'); close(gcf);
% >> text
fid = fopen(fullfile(RESULTS_DIR, ['regression_' analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt); fclose(fid);


% ===== aperio_auc =====
analysis_id = 'ASCstart_aperio_auc_r';
topos_band = aper_auc_topos;
report_txt = [];

% >> topos (all bands)
report_txt = fun_plot_topos_iv_regression_bands(topos_band, asc_meas_iv, chan_labels_63, layout_63, neighbours_63, {bands.name}, analysis_id);
% >> figure
saveas(gcf, fullfile(RESULTS_DIR, ['regression_' analysis_id '.svg']),'svg'); close(gcf);
% >> text
fid = fopen(fullfile(RESULTS_DIR, ['regression_' analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt); fclose(fid);

% ===== fooof_auc =====
analysis_id = 'ASCstart_fooof_auc_r';
topos_band = fooof_auc_topos;
report_txt = [];

% >> topos (all bands)
report_txt = fun_plot_topos_iv_regression_bands(topos_band, asc_meas_iv, chan_labels_63, layout_63, neighbours_63, {bands.name},analysis_id);
% >> figure
saveas(gcf, fullfile(RESULTS_DIR, ['regression_' analysis_id '.svg']),'svg'); close(gcf);
% >> text
fid = fopen(fullfile(RESULTS_DIR, ['regression_' analysis_id '.txt']), 'w');
fprintf(fid, '%s', report_txt); fclose(fid);
