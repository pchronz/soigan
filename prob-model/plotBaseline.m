% SERIAL
load experimentResultsSerial

% Baseline
[max_K, N] = size(baseline_correctness_serial);
% determine how many entries have actually been computed
N = max([1:N](logical(sum(baseline_correctness_serial))));
% transform the prediction hits/misses into a rates incrementally
% clustering-based
% K x N x accuracy/precision/recall/F-measure
baseline_hit_rate = zeros(max_K, N, 4);
for K = [2:max_K]
  for n = min_N:N
    % Accuracy
    baseline_hit_rate(K, n, 1) = sum(baseline_correctness_serial(K, min_N:n))/(n - min_N + 1);
    % Precision
    Z = sum(baseline_correctness_serial(K, ~logical(d(min_N:n)))) + sum(!baseline_correctness_serial(K, logical(d(min_N:n))));
    if(Z > 0)
      baseline_hit_rate(K, n, 2) = sum(baseline_correctness_serial(K, ~logical(d(min_N:n)))) / Z;
    else
      baseline_hit_rate(K, n, 2) = 0;
    endif
    % Recall
    if(sum(!d(min_N:n)) > 0)
      baseline_hit_rate(K, n, 3) = sum(baseline_correctness_serial(K, ~logical(d(min_N:n)))) / sum(!d(min_N:n));
    else
      baseline_hit_rate(K, n, 3) = 0;
    endif
    % F-measure
    Z = baseline_hit_rate(K, n, 2) + baseline_hit_rate(K, n, 3);
    if(Z > 0)
      baseline_hit_rate(K, n, 4) = 2*baseline_hit_rate(K, n, 2)*baseline_hit_rate(K, n, 3)/Z;
    else
      baseline_hit_rate(K, n, 4) = 0;
    endif
  endfor
endfor

%% hit rate - baseline
%figure()
%for K = 2:max_K
%  % total rate
%  % subplot(max_K - 1, 3, 3*(K - 2) + 1)
%  subplot(max_K - 1, 1, K - 1)
%  plot([2:N], baseline_hit_rate(K, 2:N, 1), ';Accuracy;', baseline_hit_rate(K, 2:N, 2), ';Precision;', baseline_hit_rate(K, 2:N, 3), ';Recall;', baseline_hit_rate(K, 2:N, 4), ';F-measure;');
%  ylabel(['Baseline accuracy, K = ', num2str(K)]);
%  %% 1-rate
%  %subplot(max_K - 1, 3, 3*(K - 2) + 2)
%  %plot([2:N], baseline_hit_rate(2, 2:N, 2), ';1-rate;');
%  %% 0-rate
%  %subplot(max_K - 1, 3, 3*(K - 2) + 3)
%  %plot([2:N], baseline_hit_rate(2, 2:N, 3), ';0-rate;');
%endfor
%
%% learning - baseline
%if(length(baseline_training_serial(K, :)) > N)
%  baseline_training_serial(:, N+1:end) = [];
%endif
%figure()
%for K = [2:max_K]
%  subplot(max_K - 1, 1, K - 1)
%  plot([2:N], baseline_training_serial(K, 2:N), ';Learning time;');
%  ylabel(['K = ', num2str(K)]);
%endfor
%
%% prediction - baseline
%if(length(baseline_prediction_serial(K, :)) > N)
%  baseline_prediction_serial(:, N+1:end) = [];
%endif
%figure()
%for K = [2:max_K]
%  subplot(max_K - 1, 1, K - 1)
%  plot([2:N], baseline_prediction_serial(K, 2:N), ';Prediction time;');
%  ylabel(['K = ', num2str(K)]);
%endfor

% save the processed data for plotting it in R
save baseline_hit_rate_serial.mat baseline_hit_rate 
save baseline_training_serial.mat baseline_training_serial
save baseline_prediction_serial.mat baseline_prediction_serial

