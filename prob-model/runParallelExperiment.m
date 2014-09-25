function runParallelExperiment(X, d, min_K, max_K, S = 10)
  disp('Starting the parallel experiment')
  % PARALLEL EXPERIMENT
  [D, I, N] = size(X);
  % result containers
  % (true value, predicted value) X predictions
  bernoulli_correctness_parallel = zeros(2, S*(N - floor(N/S)));
  bernoulli_training_parallel = zeros(1, S);
  bernoulli_prediction_parallel = zeros(1, S);
  % (true value, predicted value) X predictions
  svm_correctness_parallel = zeros(2, S*(N - floor(N/S)));
  svm_training_parallel = zeros(1, S);
  svm_prediction_parallel = zeros(1, S);
  % K X (true value, predicted value) X predictions
  baseline_correctness_parallel = zeros(max_K, 2, S*(N - floor(N/S)));
  baseline_training_parallel = zeros(max_K, S);
  baseline_prediction_parallel = zeros(max_K, S);

  prob_model_correctness_parallel = zeros(max_K, N);
  prob_model_training_parallel = zeros(max_K, S);
  prob_model_prediction_parallel = zeros(max_K, S);

  % Verify that we have a value for each class in the data set.
  first_neg = min([1:N](d == 0));
  first_pos = min([1:N](d == 1));
  assert(!isempty(first_neg) && !isempty(first_pos))

  % TODO
  %tic()
  %disp('Running dimensionality reduction')
  %% TODO XXX cross reduction needs to happen with each run to make it realistic.
  %[services, dims] = S);
  %X_red = extractReducedData(X, services, dims);
  %cross_red_time = toc()
  %save experimentResultsParallelCrossReductionTime.mat cross_red_time


  % Index of the current position of the tests.
  test_idx = 1;

  for s = 1:S
    [X_tr, d_tr, X_test, d_test] = splitDataCross(X, d, s, S);

    % Bernoulli
    disp('Bernoulli')
    [t_train, t_pred, correctness] = runBernoulliParallelExperiment(d_tr, d_test, s, test_idx);
    bernoulli_training_parallel(1, s) = t_train;
    bernoulli_prediction_parallel(1, s) = t_pred;
    bernoulli_correctness_parallel(:, test_idx:test_idx + length(d_test) - 1) = correctness;
    
    % SVM
    disp('SVM')
    % learn SVM
    tic()
    MODE.TYPE='rbf';
    MODE.hyperparameter.c_value=rand(1)*250;
    MODE.hyperparameter.gamma=rand(1)/10000;
    if(sum(d_tr) > 0 && sum(!d_tr) > 0)
      [X_tr_bal, d_tr_bal] = balanceData(X_tr, d_tr);
    endif
    [D, I, N_bal] = size(X_tr_bal);
    CC = train_sc(reshape(X_tr_bal, [D*I, N_bal])', (d_tr_bal + 1)', MODE);
    svm_training_parallel(1, s) = toc();
    if(CC.model.totalSV > 0)
      % predict SVM
      tic();
      hits_svm = test_sc(CC, reshape(X_test, [D*I, size(X_test)(3)])');
      hits_svm = hits_svm.classlabel - 1;
      svm_prediction_parallel(1, s) = toc();
      svm_correctness_parallel(1, test_idx:test_idx + length(d_test) - 1) = d_test;
      svm_correctness_parallel(2, test_idx:test_idx + length(d_test) - 1) = hits_svm;
      test_idx + length(d_test - 1)
    else
      warning('No support vectors during SVM training')
      svm_training_parallel(1, s) = -1;
      svm_prediction_parallel(1, s) = -1;
      svm_correctness_parallel(1, test_idx:test_idx + length(d_test) - 1) = -1;
    endif

    for K = min_K:max_K
      % TODO What happens if we balance the data set first as for the SVM?
      %X_tr = X_red(:, :, win_n:n);
      %d_tr = d(1, win_n:n);
      disp('Baseline model training --- parallel')
      tic()
      [centers, rho_base] = learnBaselineModel(K, X_tr, d_tr);
      baseline_training_parallel(K, s) = toc();
      % predict baseline
      disp('Baseline model prediction --- parallel')
      tic()
      for n = 1:size(X_test)(3)
        [p_0, p_1] = predictBaseline(X_test(:, :, n), centers, rho_base);
        baseline_correctness_parallel(K, 2, test_idx + n - 1) = double((p_0 < p_1));
      endfor
      baseline_prediction_parallel(K, s) = toc();
      baseline_correctness_parallel(K, 1, test_idx:test_idx + length(d_test) - 1) = d_test;

      %disp('Multi-mixture model training --- parallel')
      %tic()
      %[mus, Sigmas, rho, pi] = learnExactIndependent(K, X_tr, d_tr, 30);
      %elapsed = toc()
      %prob_model_training_parallel(K, n + 1) = elapsed;
      %disp('Multi-mixture model prediction --- parallel')
      %tic()
      %[p_0, p_1] = predictExactIndependent(X_red(:, :, n + 1), mus, Sigmas, rho, pi);
      %elapsed = toc()
      %prob_model_prediction_parallel(K, n + 1) = elapsed;
      %prob_model_correctness_parallel(K, n + 1) = double((p_0 < p_1) == d(n + 1));
      %prob_model_correctness_parallel(K, 1:n + 1);
    endfor

    % Update the position of our result pointer.
    test_idx += length(d_test);

    try
      save -V7 experimentResultsParallel.mat min_K max_K S bernoulli_correctness_parallel bernoulli_training_parallel bernoulli_prediction_parallel svm_correctness_parallel svm_training_parallel svm_prediction_parallel
      save -V7 experimentResultsParallelRelevantServices.mat services dims
      disp('The parallel results have been saved')
    catch
      error(last_error())
    end_try_catch

    %  endfor
    %endfor

    % Output the results
    bernoulli_correctness_parallel(:, 1:test_idx - 1)
    bernoulli_training_parallel(1, s)
    bernoulli_prediction_parallel(1, s)
    svm_correctness_parallel(:, 1:test_idx - 1)
    svm_training_parallel(1, :)
    svm_prediction_parallel(1, :)
    %baseline_correctness_parallel
    %baseline_training_parallel
    %baseline_prediction_parallel
    %prob_model_correctness_parallel
    %prob_model_training_parallel
    %prob_model_prediction_parallel
  endfor
  % Delete all unused entries in the results.
  bernoulli_correctness_parallel(:, test_idx:end) = [];
endfunction

function [t_train, t_pred, correctness] =  runBernoulliParallelExperiment(d_tr, d_test, s, test_idx)
  % Bernoulli (max likelihood)
  disp('Bernoulli model training --- parallel')
  tic()
  rho = sum(d_tr)/length(d_tr);
  t_train = toc()
  disp('Bernoulli model prediction --- parallel')
  tic()
  correctness = zeros(2, length(d_test));
  for n = 1:length(d_test)
    correctness(1, n) = d_test(n);
    % Choose a random value and compare to rho.
    correctness(2, n) = double(rand() > (1 - rho));
  endfor
  t_pred = toc()
endfunction

