
function [pks_final, locs_final] = fun_ecg_peakdetect(sig,fs,cluster_Y)

% -------------------------------
% PARAMETERS
% -------------------------------
win_sec   = 2;          % sliding window size (seconds)
step_sec  = 0.25;       % step size (seconds)
mad_mult  = 6;          % threshold multiplier
minRR_sec = 0.4;        % minimum RR distance (physiology)
consistency_perc = 0.25;

% -------------------------------
% INITIALIZE
% -------------------------------
N = length(sig);
win_samp  = round(win_sec*fs);
step_samp = round(step_sec*fs);

peak_score = zeros(1,N);   % vote counter

% -------------------------------
% SLIDING WINDOW PEAK VOTING
% -------------------------------
for start_idx = 1:step_samp:(N-win_samp)
    
    stop_idx = start_idx + win_samp - 1;
    segment  = sig(start_idx:stop_idx);
    
    % robust threshold
    seg_mad = mad(segment,1);
    thresh  = mad_mult * seg_mad;

    
    % find local peaks
    [pks_local,locs_local] = findpeaks(segment,...
        'MinPeakHeight',thresh,...
        'MinPeakDistance',round(minRR_sec*fs));
    
%     figure;
%     plot(segment); hold on; yline(thresh)    
%     plot(locs_local,pks_local,'or')
    
    % convert local indices to global indices
    locs_global = locs_local + start_idx - 1;
    
    % increment vote score
    peak_score(locs_global) = peak_score(locs_global) + 1;
end

% normalize score (optional)
max_score = max(peak_score);

% threshold based on relative consistency
consistency_thresh = consistency_perc * max_score;

candidate_locs = find(peak_score >= consistency_thresh);

locs_final = candidate_locs;
pks_final = sig(locs_final);

if cluster_Y
    min_dist = round(minRR_sec*fs);
    
    locs_clust = [];
    i = 1;
    
    while i <= length(candidate_locs)
        
        cluster = candidate_locs(i);
        
        j = i + 1;
        while j <= length(candidate_locs) && ...
                candidate_locs(j) - candidate_locs(j-1) < min_dist
            
            cluster = [cluster candidate_locs(j)];
            j = j + 1;
        end
        
        % keep the sample with highest score in cluster
        [~,idx] = max(peak_score(cluster));
        locs_clust(end+1) = cluster(idx);
        
        i = j;
    end
    
    locs_final = locs_clust;
    pks_final = sig(locs_final);
end

end