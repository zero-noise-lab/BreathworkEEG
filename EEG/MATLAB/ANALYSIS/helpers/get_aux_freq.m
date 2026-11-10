
function OUT = get_aux_freq(data_pass, nCond)

fs         = data_pass.fsample;
aux_labels = data_pass.label;

idx_pks1   = find(strcmp(aux_labels, 'PKS-1'));
idx_rrbad1 = find(strcmp(aux_labels, 'RR-BAD-1'));
idx_resp   = find(strcmp(aux_labels, 'RESP'));
idx_edr1   = find(strcmp(aux_labels, 'ecg-1-envelope'));

missing = {};
if isempty(idx_pks1),   missing{end+1} = 'PKS-1';          end
if isempty(idx_rrbad1), missing{end+1} = 'RR-BAD-1';       end
if isempty(idx_resp),   missing{end+1} = 'RESP';            end
if isempty(idx_edr1),   missing{end+1} = 'ecg-1-envelope'; end
if ~isempty(missing)
    warning('Missing physio channels: %s', strjoin(missing, ', '));
end

OUT.heart_rate    = nan(nCond, 1);
OUT.hrv_sdnn  = nan(1,nCond);
OUT.hrv_rmssd = nan(1,nCond);
OUT.hrv_pnn50 = nan(1,nCond);
OUT.resp_rate     = nan(nCond, 1);
OUT.resp_rate_std = nan(nCond, 1);
OUT.edr_amp       = nan(nCond, 1);
OUT.edr_resp_rate = nan(nCond, 1);
OUT.edr_resp_std  = nan(nCond, 1);


for c = 1:nCond

    cfg_phys        = [];
    cfg_phys.trials = c;
    data_phys       = ft_selectdata(cfg_phys, data_pass);
    sig             = data_phys.trial{1};
    nSamp           = size(sig, 2);

    % -------------------------------------------------------
    % HEART RATE + HRV from PKS-1
    % -------------------------------------------------------
    if ~isempty(idx_pks1) && ~isempty(idx_rrbad1)
        pks_sig         = sig(idx_pks1, :);
        bad_mask        = logical(sig(idx_rrbad1, :));
        peak_locs       = find(pks_sig > 0);
        peak_locs_clean = peak_locs(~bad_mask(peak_locs));
        if numel(peak_locs_clean) > 2
            rr_intervals = diff(peak_locs_clean) / fs;
            rr_clean     = rr_intervals(rr_intervals > 0.25 & rr_intervals < 2.0);
            if ~isempty(rr_clean)
                OUT.heart_rate(c) = 60 / mean(rr_clean);
            end

            % --- HRV time-domain metrics (need at least 2 clean RR
            % intervals to get a meaningful successive-difference measure)
            if numel(rr_clean) > 2
                rr_ms = rr_clean * 1000;   % seconds -> ms, standard HRV convention
                diff_rr = diff(rr_ms);

                OUT.hrv_sdnn(c)  = std(rr_ms);                                   % overall RR variability
                OUT.hrv_rmssd(c) = sqrt(mean(diff_rr.^2));                       % successive-difference variability, vagally mediated
                OUT.hrv_pnn50(c) = 100 * sum(abs(diff_rr) > 50) / numel(diff_rr); % % of successive diffs > 50ms
            end
        end
        fprintf('  cond %d | HR: %.1f bpm | SDNN: %.1f ms | RMSSD: %.1f ms | pNN50: %.1f%%\n', ...
            c, OUT.heart_rate(c), OUT.hrv_sdnn(c), OUT.hrv_rmssd(c), OUT.hrv_pnn50(c));
    end

    % -------------------------------------------------------
    % RESP RATE from RESP channel
    % -------------------------------------------------------
    if ~isempty(idx_resp)
        resp_sig       = sig(idx_resp, :);
        
%         figure; plot(resp_sig)
        
        resp_smooth = smoothdata(resp_sig, 'gaussian', fs * 5);

        valid_interval_range = [0.4 5];
        
        [~, peak_locs] = findpeaks(resp_smooth, 'MinPeakDistance', round(fs * 0.3),'MinPeakHeight', median(resp_sig)*3);
        
        n_peaks = numel(peak_locs);

%         pksig = nan(length(resp_sig),1);
%         pksig(peak_locs) = 0;
%         figure; plot(resp_smooth); hold on; plot(pksig,'ok');    
        
        if n_peaks > 2
            
                 duration_min = (peak_locs(end) - peak_locs(1)) / fs / 60;
        fprintf('detected %d peaks over %.1f min (%.1f bpm by raw count)\n', ...
            n_peaks, duration_min, n_peaks/duration_min);
   
            
            
            intervals       = diff(peak_locs) / fs;
            intervals_clean = intervals(~isoutlier(intervals));
            if ~isempty(intervals_clean)
                OUT.resp_rate(c)     = 60 / mean(intervals_clean);
                OUT.resp_rate_std(c) = std(intervals_clean) * 60 / mean(intervals_clean)^2;
            end
        end
        fprintf('  cond %d | RESP: %.1f bpm\n', ...
            c, OUT.resp_rate(c));

        
    end

    % -------------------------------------------------------
    % EDR AMPLITUDE + RESP RATE from ecg-1-envelope
    % -------------------------------------------------------
    if ~isempty(idx_edr1) && ~isempty(idx_rrbad1)
        edr_sig           = sig(idx_edr1, :);
        bad_mask          = logical(sig(idx_rrbad1, :));
        edr_sig(bad_mask) = NaN;

        OUT.edr_amp(c) = mean(edr_sig, 'omitnan');
        
%         valid_resp_interval_range = [0.4 8];
     
        edr_smooth = smoothdata(edr_sig, 'gaussian', fs * 5);

        [~, peak_locs] = findpeaks(edr_smooth, 'MinPeakDistance', round(fs * 0.3));
%         
%         pksig = nan(length(edr_smooth),1);
%         pksig(peak_locs) = max(edr_sig);
%         figure; plot(edr_sig); hold on; plot(edr_smooth);  plot(pksig,'ok');

        if numel(peak_locs) > 2
            intervals       = diff(peak_locs) / fs;
%             intervals_clean = intervals(intervals > valid_resp_interval_range(1) & intervals < valid_resp_interval_range(2));

% % reject outliers relative to THIS condition's own intervals, not a
% % guessed global window
% med     = median(intervals);
% mad_val = mad(intervals, 1);     % median absolute deviation
% k       = 4;                      % how many MADs out counts as junk, tune by eye
% intervals_clean = intervals(intervals > med - k*mad_val & intervals < med + k*mad_val);

intervals_clean = filloutliers(intervals,nan);

            if ~isempty(intervals_clean)
                OUT.edr_resp_rate(c) = 60 / nanmean(intervals_clean);
                OUT.edr_resp_std(c)  = nanstd(intervals_clean) * 60 / nanmean(intervals_clean)^2;
            end
        end
        fprintf('  cond %d | EDR amp: %.3f | EDR resp: %.1f bpm \n', ...
            c, OUT.edr_amp(c), OUT.edr_resp_rate(c));
    end

end
end