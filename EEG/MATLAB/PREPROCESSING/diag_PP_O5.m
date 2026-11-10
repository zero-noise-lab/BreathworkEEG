% ======= PP05 DIAGNOSTIC ANALYSIS =======
clear all; clc;
DIAGPATH = 'PATH_TO_SAVEPATH';

diagfiles = dir(fullfile(DIAGPATH, '*diag.mat'));
fprintf('Found %d diagnostic files\n\n', numel(diagfiles));

% ======= COLLECT =======
subject           = {};
n_chans_interp    = [];
badchans_beforeica      = {};
badchans_afterica = {};

for i = 1:numel(diagfiles)
    d = load(fullfile(diagfiles(i).folder, diagfiles(i).name));
    w = d.diagOut;

    subject{end+1}           = w.subject;
%     n_chans_interp(end+1)    = w.n_chans_interp;
    badchans_beforeica{end+1}      = strjoin(w.badchanLabels_beforeICA,     ', ');
    badchans_afterica{end+1} = strjoin(w.badchanLabels_afterICA,', ');
    n_chans_interp(end+1)  = numel(w.badchanLabels_beforeICA) + numel(w.badchanLabels_afterICA) ;
end

% ======= BUILD TABLE =======
T = table( ...
    subject(:), ...
    n_chans_interp(:), ...
    badchans_beforeica(:), ...
    badchans_afterica(:), ...
    'VariableNames', { ...
        'Subject', ...
        'nChanInterp', ...
        'BadChans_BeforeICA', ...
        'BadChans_AfterICA'});

disp(T)

save(T, fullfile(DIAGPATH, 'diag_summary.mat'));
