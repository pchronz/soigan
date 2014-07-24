function dims = crossReduceDimensions(X, d)
  S = 10;
  % Less-frequent target value.
  d_less = 1;
  if(sum(d) > (length(d)/2))
    d_less = 0;
  endif
  % Get 
  % Run an S-fold cross-validation, reducing the dimensions each time. 
  % Separate the data based on the less-frequent taret values.
  % Use the union from all runs as the required dimensionality.

  % XXX Quick fix: just run the dimensionality reduction with random data splitting multiple times and use union to get the final dimensions.
  dims = [];
  for i =1:10
    dims = union(dims, reduceDimensions(X, d));
    dims
  endfor
  disp('Keeping...')
  dims
end

function dims = reduceDimensions(X, d)
  [D, I, N] = size(X);

  % Quality metrics for each iteration. The first value corresponds to the full data set.
  acc = zeros(I, 1);
  prec = zeros(I, 1);
  rec = zeros(I, 1);
  F = zeros(I, 1);

  % Split the data into training and test set.
  [X_tr, d_tr, X_test, d_test] = splitData(X, d, 0.8);
  % Balance the training set.
  [X_tr, d_tr] = balanceData(X_tr, d_tr);
  % Evaluate the SVM: train, test, and calculate the metrics.
  [accuracy, precision, recall, F_measure] = evalSvm(X_tr, d_tr, X_test, d_test);
  acc(1) = accuracy;
  prec(1) = precision;
  rec(1) = recall;
  F(1) = F_measure;

  % Leave out services, one by one.
  % Remaining services for each iteration.
  Is = true(I, I);
  for i = 1:I - 1
    disp(['Removing the ' num2str(i) 'th service now'])
    % Get the indices of the remaining services.
    idx = (1:I)(Is(i + 1, :));
    % List of quality metrics for each remaining service.
    acc_rem = zeros(length(idx), 1);
    prec_rem = zeros(length(idx), 1);
    rec_rem = zeros(length(idx), 1);
    F_rem = zeros(length(idx), 1);
    % Iterate over the remaining services.
    for i_rem = idx
      % Prepare the data.
      % Evaluate the SVM: train, test, and calculate the metrics.
      [accuracy, precision, recall, F_measure] = evalSvm(X_tr(:, idx(idx != i_rem), :), d_tr, X_test(:, idx(idx != i_rem), :), d_test);
      % Store the quality metrics.
      acc_rem((1:length(idx))(idx == i_rem)) = accuracy;
      prec_rem((1:length(idx))(idx == i_rem)) = precision;
      rec_rem((1:length(idx))(idx == i_rem)) = recall;
      F_rem((1:length(idx))(idx == i_rem)) = F_measure;
    endfor
    % Determine the least contributing service.
    % TODO Use an objective function that combines all of the separate metrics.
    [max_q, max_ix] = max(F_rem);
    % Remove the least contributing service.
    Is(i + 1:end, idx(max_ix)) = false;
    % Store the results for the best sub-set.
    acc(i + 1) = acc_rem(max_ix);
    prec(i + 1) = prec_rem(max_ix);
    rec(i + 1) = rec_rem(max_ix);
    F(i + 1) = F_rem(max_ix);
  endfor

  % Determine how deep you are willing to go.
  % TODO Use a combined objective function.
  % First get the depth with the best result.
  [q_best, d_best] = max(F);
  % Get the depths that are greater than d_best at which the results are acceptable.
  q_min = q_best*(1 - 1/20);
  F_ = F(d_best:end);
  % Choose the greatest acceptable detph.
  d_deepest = max((d_best:I)(F_ >= q_min));
  % Get the sub-set of the maximum depth.
  dims = (1:I)(Is(d_deepest, :));
  disp(['Removing ' num2str(d_deepest - 1) ' out of ' num2str(I) ' services'])
endfunction

function [X_tr, d_tr, X_test, d_test] = splitData(X, d, ratio)
  [D, I, N] = size(X);
  % split the data into training and validation sets
  % first split the data by target classes
  flags_0 = (d == 0);
  flags_1 = !flags_0;
  idx_0 = (1:N)(flags_0);
  idx_1 = (1:N)(flags_1);
  % 0s first
  idx = 0;
  while(sum(idx) == 0)
    idx = unidrnd(10, 1, sum(flags_0));
    idx = idx < 10*ratio;
  endwhile
  idx_0_test = idx_0(idx);
  idx_0_train = idx_0(!idx);
  % 1s second
  idx = 0;
  while(sum(idx) == 0)
    idx = unidrnd(10, 1, sum(flags_1));
    idx = idx < 8;
  endwhile
  idx_1_test = idx_1(idx);
  idx_1_train = idx_1(!idx);
  % now assemble both into common training and test sets
  X_test = X(:, :, [idx_0_test, idx_1_test]);
  d_test = d([idx_0_test, idx_1_test]);
  X_tr = X(:, :, [idx_0_train, idx_1_train]);
  d_tr = d([idx_0_train, idx_1_train]);
endfunction

function CC = trainSvm(X_tr, d_tr)
  MODE.TYPE='rbf';
  MODE.hyperparameter.c_value=0.5*250;
  MODE.hyperparameter.gamma=0.5/10000;
  [D, I, N_tr] = size(X_tr);
  CC = train_sc(reshape(X_tr, [D*I, N_tr])', (d_tr + 1)', MODE);
endfunction

function hits_svm = testSvm(X_test, d_test, CC)
  [D, I, N_te] = size(X_test);
  hits_svm = test_sc(CC, reshape(X_test, [D*I, N_te])');
  hits_svm = hits_svm.classlabel - 1;
  hits_svm = hits_svm == d_test;
endfunction

function [accuracy, precision, recall, F_measure] = calculateQuality(hits_svm, d_test)
  tp = sum(hits_svm .* d_test);
  fp = sum((!hits_svm) .* (!d_test));
  % Accuracy
  accuracy = sum(hits_svm)/length(hits_svm);
  % Precision
  precision = tp/(tp + fp);
  % Recall
  recall = tp/sum(d_test);
  % F-Measure
  F_measure = 2*precision*recall/(precision + recall);
endfunction

function [accuracy, precision, recall, F_measure] = evalSvm(X_tr, d_tr, X_test, d_test)
  % Train the SVM with all dimensions to obtain a baseline value.
  CC = trainSvm(X_tr, d_tr);
  % Test the SVM on the test set.
  hits_svm = testSvm(X_test, d_test, CC);
  % Compute the baseline quality metrics: accuracy, precision, recall, F-measure.
  [accuracy, precision, recall, F_measure] = calculateQuality(hits_svm, d_test);
endfunction

