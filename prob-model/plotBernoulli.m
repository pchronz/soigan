% SERIAL
load experimentResultsSerial

N = size(bernoulli_correctness_serial');
% determine how many entries have actually been computed
N = max([1:N](logical(sum(baseline_correctness_serial))));
% Bernoulli
% total rate/precision/recall/F-measure
bernoulli_hit_rate = zeros(4, N);
for n = min_N:N
  % get the values that count at all
  % count the hits
  bernoulli_hits = bernoulli_correctness_serial(min_N:n);
  % Accuracy
  bernoulli_hit_rate(1, n) = sum(bernoulli_hits)/(n - 1);
  % Precision
  Z = sum(bernoulli_hits .* !d(min_N:n)) + sum(!bernoulli_hits(logical(d(min_N:n))));
  if(Z > 0)
    bernoulli_hit_rate(2, n) = sum(bernoulli_hits .* !d(min_N:n))/Z;
  else
    bernoulli_hit_rate(2, n) = 0;
  endif
  % Recall
  Z = sum(bernoulli_hits .* !d(min_N:n)) + sum(!bernoulli_hits(logical(!d(min_N:n))));
  if(Z > 0)
    bernoulli_hit_rate(3, n) = sum(bernoulli_hits .* !d(min_N:n))/Z;
  else
    bernoulli_hit_rate(3, n) = 0;
  endif
  % F-measure
  Z = bernoulli_hit_rate(2, n) + bernoulli_hit_rate(3, n);
  if(Z > 0)
    bernoulli_hit_rate(4, n) = 2*bernoulli_hit_rate(2, n)*bernoulli_hit_rate(3, n)/Z;
  endif
endfor

%% hit rate - Bernoulli
%figure()
%plot([2:N], bernoulli_hit_rate(1, 2:N), ';Accuracy;', bernoulli_hit_rate(2, 2:N), ';Precision;', bernoulli_hit_rate(3, 2:N), ';Recall;', bernoulli_hit_rate(4, 2:N), ';F-measure;');
%ylabel('Bernoulli accuracy');
%
%% learning - Bernoulli
%if(length(bernoulli_training_serial) > N)
%  bernoulli_training_serial(N+1:end) = [];
%endif
%figure()
%plot([2:N], bernoulli_training_serial(2:N), ';Learning time;');
%ylabel('Bernoulli');
%
%% prediction - Bernoulli
%if(length(bernoulli_prediction_serial) > N)
%  bernoulli_prediction_serial(N+1:end) = [];
%endif
%figure()
%plot([2:N], bernoulli_prediction_serial(2:N), ';Prediction time;');
%ylabel('Bernoulli');

% save the processed data for plotting it in R
save bernoulli_hit_rate.mat bernoulli_hit_rate
save bernoulli_training_serial.mat bernoulli_training_serial
save bernoulli_prediction_serial.mat bernoulli_prediction_serial

