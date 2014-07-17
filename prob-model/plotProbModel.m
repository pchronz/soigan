% SERIAL
load experimentResultsSerial

% Prob model
[max_K, N] = size(prob_model_correctness_serial);
% determine how many entries have actually been computed
N = max([1:N](logical(sum(prob_model_correctness_serial))));
% transform the prediction hits/misses into a rates incrementally
% clustering-based
% K x N x accuracy/precision/recall/F-measure
prob_model_hit_rate = zeros(max_K, N, 4);
for K = [2:max_K]
  for n = min_N:N
    % Accuracy
    prob_model_hit_rate(K, n, 1) = sum(prob_model_correctness_serial(K, min_N:n))/(n - min_N + 1);
    % Precision
    Z = sum(prob_model_correctness_serial(K, logical(d(min_N:n)))) + sum((!prob_model_correctness_serial(K, min_N:n)) .* !d(min_N:n));
    if(Z > 0)
      prob_model_hit_rate(K, n, 2) = sum(prob_model_correctness_serial(K, logical(d(min_N:n)))) / Z;
    else
      prob_model_hit_rate(K, n, 2) = 0;
    endif
    % Recall
    if(sum(d(min_N:n)) > 0)
      prob_model_hit_rate(K, n, 3) = sum(prob_model_correctness_serial(K, logical(d(min_N:n)))) / sum(d(min_N:n));
    else
      prob_model_hit_rate(K, n, 3) = 0;
    endif
    % F-measure
    Z = prob_model_hit_rate(K, n, 2) + prob_model_hit_rate(K, n, 3);
    if(Z > 0)
      prob_model_hit_rate(K, n, 4) = 2*prob_model_hit_rate(K, n, 2)*prob_model_hit_rate(K, n, 3)/Z;
    else
      prob_model_hit_rate(K, n, 4) = 0;
    endif
  endfor
endfor

%% hit rate - prob_model
%figure()
%for K = 2:max_K
%  % total rate
%  % subplot(max_K - 1, 3, 3*(K - 2) + 1)
%  subplot(max_K - 1, 1, K - 1)
%  plot([2:N], prob_model_hit_rate(K, 2:N, 1), ';Accuracy;', prob_model_hit_rate(K, 2:N, 2), ';Precision;', prob_model_hit_rate(K, 2:N, 3), ';Recall;', prob_model_hit_rate(K, 2:N, 4), ';F-measure;');
%  ylabel(['Baseline accuracy, K = ', num2str(K)]);
%  %% 1-rate
%  %subplot(max_K - 1, 3, 3*(K - 2) + 2)
%  %plot([2:N], prob_model_hit_rate(2, 2:N, 2), ';1-rate;');
%  %% 0-rate
%  %subplot(max_K - 1, 3, 3*(K - 2) + 3)
%  %plot([2:N], prob_model_hit_rate(2, 2:N, 3), ';0-rate;');
%endfor
%
%% learning - prob_model
%if(length(prob_model_training_serial(K, :)) > N)
%  prob_model_training_serial(:, N+1:end) = [];
%endif
%figure()
%for K = [2:max_K]
%  subplot(max_K - 1, 1, K - 1)
%  plot([2:N], prob_model_training_serial(K, 2:N), ';Learning time;');
%  ylabel(['K = ', num2str(K)]);
%endfor
%
%% prediction - prob_model
%if(length(prob_model_prediction_serial(K, :)) > N)
%  prob_model_prediction_serial(:, N+1:end) = [];
%endif
%figure()
%for K = [2:max_K]
%  subplot(max_K - 1, 1, K - 1)
%  plot([2:N], prob_model_prediction_serial(K, 2:N), ';Prediction time;');
%  ylabel(['K = ', num2str(K)]);
%endfor

% save the processed data for plotting it in R
save prob_model_hit_rate_serial.mat prob_model_hit_rate 
save prob_model_training_serial.mat prob_model_training_serial
save prob_model_prediction_serial.mat prob_model_prediction_serial

