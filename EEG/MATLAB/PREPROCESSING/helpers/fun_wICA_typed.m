
function [cleaned_data, wicaDiag] = fun_wICA_typed(EEG, icTypes)
% FUN_WICA_TYPED  Apply typed component cleaning.
%
%   [cleaned_data, wicaDiag] = fun_wICA_typed(EEG, icTypes)
%
% INPUTS
%   EEG     - EEGLAB struct with ICA already run (EEG.icaweights etc.)
%   icTypes - n_comps x 1 cell array of 'brain' | 'eye' | 'muscle' | 'ambiguous'
%             as produced by get_component_types.m (with any manual
%             overrides already applied)
%
% ACTIONS PER TYPE
%   'brain'     -> kept untouched
%   'eye'       -> fully removed (zeroed before back-projection)
%   'muscle'    -> fully removed (zeroed before back-projection)
%   'ambiguous' -> wavelet-thresholded (db4, per-level universal threshold),
%                  i.e. the old wICA denoising behaviour, reserved now for
%                  components where evidence didn't clearly resolve to a
%                  single class

disp('Starting typed W-ICA cleaning (db4)...');
if isempty(EEG.icaweights)
    error('ICA has not been run.');
end
if nargin < 2 || isempty(icTypes)
    error('icTypes must be provided (see get_component_types.m).');
end

wavelet_family = 'db4';
decomp_level   = 5;

ica_activations                   = eeg_getdatact(EEG, 'component', 1:size(EEG.icaweights,1));
[num_components, original_length] = size(ica_activations);

if numel(icTypes) ~= num_components
    error('icTypes has %d entries but EEG has %d components.', numel(icTypes), num_components);
end

wicaDiag                 = struct();
wicaDiag.icTypes         = icTypes;
wicaDiag.comp_removed    = false(num_components,1);   % eye or muscle
wicaDiag.comp_kept_brain = false(num_components,1);
wicaDiag.comp_smoothed   = false(num_components,1);   % ambiguous

cleaned_activations = zeros(size(ica_activations));

divisor        = 2^decomp_level;
new_length     = ceil(original_length / divisor) * divisor;
padding_length = new_length - original_length;

fprintf('Processing %d components...\n', num_components);
for ic = 1:num_components
    switch icTypes{ic}
        case {'eye','muscle','cardiac'}
            cleaned_activations(ic,:) = 0;
            wicaDiag.comp_removed(ic) = true;
            fprintf('  IC %d: removed (%s)\n', ic, icTypes{ic});

        case 'brain'
            cleaned_activations(ic,:) = ica_activations(ic,:);
            wicaDiag.comp_kept_brain(ic) = true;

        case 'ambiguous'
            component_signal = ica_activations(ic,:);
            padded_signal = wextend('1D', 'sym', component_signal, padding_length, 'r');
            padded_length = length(padded_signal);
            [swa, swd] = swt(padded_signal, decomp_level, wavelet_family);
            for lev = 1:decomp_level
                sigma      = mad(swd(lev,:), 1) / 0.6745;
                thr        = sqrt(2 * log(padded_length)) * sigma;
                swd(lev,:) = wthresh(swd(lev,:), 's', thr);
            end
            cleaned_padded            = iswt(swa, swd, wavelet_family);
            cleaned_activations(ic,:) = cleaned_padded(1:original_length);
            wicaDiag.comp_smoothed(ic) = true;

        otherwise
            error('Unrecognized IC type "%s" for component %d.', icTypes{ic}, ic);
    end
end

fprintf('Typed W-ICA complete.\n');
fprintf('  Removed (eye+cardiac+muscle): %d\n', sum(wicaDiag.comp_removed));
fprintf('  Brain kept untouched: %d\n', sum(wicaDiag.comp_kept_brain));
fprintf('  Wavelet smoothed:     %d\n', sum(wicaDiag.comp_smoothed));

% Subtract only the artifact contribution, preserve residual (FIXED ON ADAPTED SCRIPT!!)
cleaned_data = EEG.data - EEG.icawinv * (ica_activations - cleaned_activations);

end
