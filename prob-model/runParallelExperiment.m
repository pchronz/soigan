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

  disp('Running dimensionality reduction')
  tic()
  [services, dims] = crossReduceDimensions(X, d, 4);
  X_red = extractReducedData(X, services, dims);
  cross_red_time = toc()
  save experimentResultsParallelCrossReductionTime.mat cross_red_time


  % Index of the current position of the tests.
  test_idx = 1;

  for s = 1:S
    [X_tr, d_tr, X_test, d_test] = splitDataRand(X_red, d, 0.5);

    % Bernoulli
    disp('Bernoulli')
    [t_train, t_pred, correctness] = runBernoulliParallelExperiment(d_tr, d_test);
    bernoulli_training_parallel(1, s) = t_train;
    bernoulli_prediction_parallel(1, s) = t_pred;
    bernoulli_correctness_parallel(:, test_idx:test_idx + length(d_test) - 1) = correctness;
    
    % SVM
    disp('SVM')
    [t_train, t_pred, correctness] = runSvmParallelExperiment(X_tr, d_tr, X_test, d_test);
    svm_training_parallel(1, s) = t_train;
    svm_prediction_parallel(1, s) = t_pred;
    svm_correctness_parallel(:, test_idx:test_idx + length(d_test) - 1) = correctness;

    % Approximate Model
    disp('Approximate Model')
    [t_train, t_pred, correctness] = runApproximateParallelExperiment(X_tr, d_tr, X_test, d_test, min_K, max_K);
    baseline_training_parallel(:, s) = t_train;
    baseline_prediction_parallel(:, s) = t_pred;
    baseline_correctness_parallel(:, :, test_idx:test_idx + length(d_test) - 1) = correctness;

    for K = min_K:max_K
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
      save -V7 experimentResultsParallel.mat min_K max_K S bernoulli_correctness_parallel bernoulli_training_parallel bernoulli_prediction_parallel svm_correctness_parallel svm_training_parallel svm_prediction_parallel baseline_correctness_parallel baseline_training_parallel baseline_prediction_parallel
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
    baseline_correctness_parallel(:, :, 1:test_idx - 1)
    baseline_training_parallel(1, s)
    baseline_prediction_parallel(1, s)
    %prob_model_correctness_parallel
    %prob_model_training_parallel
    %prob_model_prediction_parallel
  endfor
  % Delete all unused entries in the results.
  bernoulli_correctness_parallel(:, test_idx:end) = [];
  svm_correctness_parallel(:, test_idx:end) = [];
  baseline_correctness_parallel(:, :, test_idx:end) = [];

  % Final, trimmed save
  try
    save -V7 experimentResultsParallel.mat min_K max_K S bernoulli_correctness_parallel bernoulli_training_parallel bernoulli_prediction_parallel svm_correctness_parallel svm_training_parallel svm_prediction_parallel baseline_correctness_parallel baseline_training_parallel baseline_prediction_parallel
    save -V7 experimentResultsParallelRelevantServices.mat services dims
    disp('The parallel results have been saved')
  catch
    error(last_error())
  end_try_catch
endfunction

function [t_train, t_pred, correctness] =  runBernoulliParallelExperiment(d_tr, d_test)
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

function [t_train, t_pred, correctness] =  runSvmParallelExperiment(X_tr, d_tr, X_test, d_test)
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
    t_train = toc();
    correctness = zeros(2, length(d_test))
    correctness(1, :) = d_test;
    if(CC.model.totalSV > 0)
      % predict SVM
      tic();
      hits_svm = test_sc(CC, reshape(X_test, [D*I, size(X_test)(3)])');
      hits_svm = hits_svm.classlabel - 1;
      t_pred = toc();
      correctness(2, :) = hits_svm;
    else
      error('No support vectors during SVM training')
      t_train = -1;
      t_pred = -1;
      correctness(1, 1:length(d_test) - 1) = -1;
    endif
endfunction

function [t_train, t_pred, correctness] =  runApproximateParallelExperiment(X_tr, d_tr, X_test, d_test, min_K, max_K)
  t_train = zeros(max_K, 1);
  t_pred = zeros(max_K, 1);
  correctness = zeros(max_K, 2, length(d_test));
  for K = min_K:max_K
    disp('Baseline model training --- parallel')
    tic()
    [centers, rho_base] = learnBaselineModel(K, X_tr, d_tr);
    t_train(K) = toc();
    % predict baseline
    disp('Baseline model prediction --- parallel')
    tic()
    for n = 1:length(d_test)
      [p_0, p_1] = predictBaseline(X_test(:, :, n), centers, rho_base);
      correctness(K, 2, n) = double((p_0 < p_1));
    endfor
    t_pred(K) = toc();
    correctness(K, 1, 1:length(d_test)) = d_test;
  endfor
endfunction

