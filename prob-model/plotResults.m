% SERIAL
load experimentResultsSerial
[max_K, N] = size(baseline_correctness_serial);
% transform the prediction hits/misses into a rates incrementally
% clustering-based
% K x N x total rate/1-rate/0-rate
baseline_hit_rates = zeros(max_K, N, 3);
for K = [2:max_K]
  for n = 2:N
    baseline_hit_rate(K, n, 1) = sum(baseline_correctness_serial(K, 2:n))/(n - 1);
    baseline_hit_rate(K, n, 2) = sum(baseline_correctness_serial(2, logical(d(2:n)))) / sum(d(2:n));
    baseline_hit_rate(K, n, 3) = sum(baseline_correctness_serial(2, ~logical(d(2:n)))) / sum(!d(2:n));
  endfor
endfor
% SVM
% total rate/1-rate/0-rate
svm_hit_rate = zeros(3, N);
for n = 2:N
  % get the values that count at all
  valid_idxs = svm_correctness_serial(2:n) > -1;
  % count the hits
  svm_hits = svm_correctness_serial(2:n) .* double(valid_idxs);
  if(sum(double(valid_idxs)) > 0)
    svm_hit_rate(1, n) = sum(svm_hits) / sum(double(valid_idxs));
    svm_hit_rate(2, n) = sum(svm_hits .* d(2:n))/ sum(d(2:n));
    svm_hit_rate(3, n) = sum(svm_hits .* !d(2:n))/ sum(!d(2:n));
  else
    svm_hit_rate(1, n) = -1;
    svm_hit_rate(2, n) = -1;
    svm_hit_rate(3, n) = -1;
  endif
endfor
% hit rate - baseline
figure()
for K = 2:max_K
  % total rate
  % subplot(max_K - 1, 3, 3*(K - 2) + 1)
  subplot(max_K - 1, 1, K - 1)
  plot([2:N], baseline_hit_rate(2, 2:end, 1), ';Total rate;', baseline_hit_rate(2, 2:end, 2), ';1-rate;', baseline_hit_rate(2, 2:end, 3), ';0-rate;');
  ylabel(['Baseline accuracy, K = ', num2str(K)]);
  %% 1-rate
  %subplot(max_K - 1, 3, 3*(K - 2) + 2)
  %plot([2:N], baseline_hit_rate(2, 2:end, 2), ';1-rate;');
  %% 0-rate
  %subplot(max_K - 1, 3, 3*(K - 2) + 3)
  %plot([2:N], baseline_hit_rate(2, 2:end, 3), ';0-rate;');
endfor
% hit rate - SVM
figure()
plot([2:N], svm_hit_rate(1, 2:end), ';Total rate;', svm_hit_rate(2, 2:end), ';1-rate;', svm_hit_rate(3, 2:end), ';0-rate;');
ylabel('SVM accuracy');
% learning - baseline
figure()
for K = [2:max_K]
  subplot(max_K - 1, 1, K - 1)
  plot([2:N], baseline_training_serial(2, 2:end), ';Learning time;');
  ylabel(['K = ', num2str(K)]);
endfor
% learning - SVM
figure()
plot([2:N], svm_training_serial(2:end), ';Learning time;');
% prediction - baseline
figure()
for K = [2:max_K]
  subplot(max_K - 1, 1, K - 1)
  plot([2:N], baseline_prediction_serial(2, 2:end), ';Prediction time;');
  ylabel(['K = ', num2str(K)]);
endfor
% prediction - SVM
figure()
plot([2:N], svm_prediction_serial(2:end), ';Prediction time;');
ylabel('SVM');

