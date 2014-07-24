% SERIAL
load experimentResultsSerial

N = size(svm_correctness_serial');
% determine how many entries have actually been computed

% SVM
% total rate/precision/recall/F-measure
svm_hit_rate = zeros(4, N);
for n = min_N:N
  % get the values that count at all
  valid_idxs = svm_correctness_serial(min_N:n) > -1;
  % count the hits
  svm_hits = svm_correctness_serial(min_N:n) .* double(valid_idxs);
  if(sum(double(valid_idxs)) > 0)
    % Accuracy
    svm_hit_rate(1, n) = sum(svm_hits)/(n - min_N + 1);
    %svm_hit_rate(1, n) = sum(svm_hits)/sum(double(valid_idxs));
    % Precision
    Z = sum(svm_hits .* d(min_N:n)) + sum((!svm_hits) .* (!d(min_N:n)));
    if(Z > 0)
      svm_hit_rate(2, n) = sum(svm_hits .* d(min_N:n))/Z;
    else
      svm_hit_rate(2, n) = 0;
    endif
    % Recall
    Z = sum(d(min_N:n));
    if(Z > 0)
      svm_hit_rate(3, n) = sum(svm_hits .* d(min_N:n))/Z;
    else
      svm_hit_rate(3, n) = 0;
    endif
    % F-measure
    Z = svm_hit_rate(2, n) + svm_hit_rate(3, n);
    if(Z > 0)
      svm_hit_rate(4, n) = 2*svm_hit_rate(2, n)*svm_hit_rate(3, n)/Z;
    endif
  else
    svm_hit_rate(1, n) = 0;
    svm_hit_rate(2, n) = 0;
    svm_hit_rate(3, n) = 0;
    svm_hit_rate(4, n) = 0;
  endif
endfor

%% hit rate - SVM
%figure()
%plot([2:N], svm_hit_rate(1, 2:N), ';Accuracy;', svm_hit_rate(2, 2:N), ';Precision;', svm_hit_rate(3, 2:N), ';Recall;', svm_hit_rate(4, 2:N), ';F-measure;');
%ylabel('SVM accuracy');
%
%% learning - SVM
%if(length(svm_training_serial) > N)
%  svm_training_serial(N+1:end) = [];
%endif
%% set the "invalid" values to 0
%svm_training_serial(svm_training_serial == -1) = 0;
%figure()
%plot([2:N], svm_training_serial(2:N), ';Learning time;');
%ylabel('SVM');
%
%% prediction - SVM
%if(length(svm_prediction_serial) > N)
%  svm_prediction_serial(N+1:end) = [];
%endif
%% set the "invalid" values to 0
%svm_prediction_serial(svm_prediction_serial == -1) = 0;
%figure()
%plot([2:N], svm_prediction_serial(2:N), ';Prediction time;');
%ylabel('SVM');

% save the processed data for plotting it in R
save svm_hit_rate.mat svm_hit_rate
save svm_training_serial.mat svm_training_serial
save svm_prediction_serial.mat svm_prediction_serial

