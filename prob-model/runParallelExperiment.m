function runParallelExperiment(X, d, min_K, max_K, S = 10)
  disp('Starting the parallel experiment')
  % PARALLEL EXPERIMENT
  [D, I, N] = size(X);
  % result containers
  baseline_correctness_parallel = zeros(max_K, N);
  % TODO Store the amount of data points in each and the value of S
  baseline_training_parallel = zeros(max_K, S);
  baseline_prediction_parallel = zeros(max_K, S);
  prob_model_correctness_parallel = zeros(max_K, N);
  prob_model_training_parallel = zeros(max_K, S);
  prob_model_prediction_parallel = zeros(max_K, S);
  svm_correctness_parallel = zeros(1, N);
  svm_training_parallel = zeros(1, S);
  svm_prediction_parallel = zeros(1, S);
  % (true value, predicted value) X predictions
  bernoulli_correctness_parallel = zeros(2, S*(N - floor(N/S)));
  bernoulli_training_parallel = zeros(1, S);
  bernoulli_prediction_parallel = zeros(1, S);

  % Verify that we have a value for each class in the data set.
  first_neg = min([1:N](d == 0));
  first_pos = min([1:N](d == 1));
  assert(!isempty(first_neg) && !isempty(first_pos))

  % Index of the current position of the tests.
  test_idx = 1;

  for s = 1:S
    [X_test, d_test, X_tr, d_tr] = splitDataCross(X, d, s, S);

    [t_train, t_pred, correctness] = runBernoulliParallelExperiment(d_tr, d_test, s, test_idx);
    bernoulli_training_parallel(1, s) = t_train;
    bernoulli_prediction_parallel(1, s) = t_pred;
    bernoulli_correctness_parallel(:, test_idx:test_idx + length(d_test) - 1) = correctness;
    
    %last_training = 0;
    %% SVM
    %for n = min_N:N - 1
    %  n

    %  win_n = max(1, n - win_len + 1);
    %  X_tr = X(:, :, win_n:n);
    %  d_tr = d(1, win_n:n);
    %  if(last_training == 0 || (n - last_training) >= refresh_rate)
    %    disp('SVM training')
    %    % learn SVM
    %    tic()
    %    MODE.TYPE='rbf';
    %    MODE.hyperparameter.c_value=rand(1)*250;
    %    MODE.hyperparameter.gamma=rand(1)/10000;
    %    if(sum(d_tr) > 0 && sum(!d_tr) > 0)
    %      [X_tr, d_tr] = balanceData(X_tr, d_tr);
    %    endif
    %    [D, I, N] = size(X_tr);
    %    CC = train_sc(reshape(X_tr, [D*I, N])', (d_tr + 1)', MODE);
    %    % XXX why does the SVM not work below a certain number of training vectors?
    %    % ==> cause it ain't got more than one class observed!
    %    elapsed = toc();
    %    svm_training_parallel(n) = elapsed;
    %    last_training = n;
    %  endif
    %  if(CC.model.totalSV > 0)
    %    disp('SVM prediction')
    %    % predict SVM
    %    tic();
    %    [D, I, N_te] = size(X(:, :, n + 1));
    %    hits_svm = test_sc(CC, reshape(X(:, :, n + 1), [D*I, N_te])');
    %    hits_svm = hits_svm.classlabel - 1;
    %    hits_svm = sum(hits_svm == d(1, n + 1));
    %    elapsed = toc();
    %    svm_prediction_parallel(n) = elapsed;
    %    assert(hits_svm == 0 || hits_svm == 1);
    %    svm_correctness_parallel(n) = hits_svm;
    %  else
    %    % disp('SVM model failed at...')
    %    % n
    %    svm_training_parallel(n) = -1;
    %    svm_correctness_parallel(n) = -1;
    %    svm_prediction_parallel(n) = -1;
    %  endif
    %endfor
    %[D, I, N] = size(X);

    %% going one step ahead is fine as long as the time required for prediction and training
    %% is less than the current time plus the time for which we wish to predict. Actually there
    %% we also need to count in a slack that is the time until which the observed global value
    %% (via Cern) will be available and the time that we need to react and fix an error (MTTR).
    %% baseline & prob model
    %%for n = max_K:N - 1
    %last_training = 0;
    %tic()
    %disp('Running dimensionality reduction')
    %% TODO XXX cross reduction needs to happen with each run to make it realistic.
    %[services, dims] = S);
    %X_red = extractReducedData(X, services, dims);
    %cross_red_time = toc()
    %save experimentResultsParallelCrossReductionTime.mat cross_red_time
    %for n = min_N:N - 1
    %  for K = min_K:max_K
    %    disp('n -- parallel')
    %    disp([num2str(n), '/', num2str(N - 1), ' = ', num2str(n/N)*100, '%'])
    %    disp('K -- parallel')
    %    disp([num2str(K), '/', num2str(max_K), ' = ', num2str(K/max_K)*100, '%'])
    %    % TODO What happens if we balance the data set first as for the SVM?
    %    win_n = max(1, n - win_len + 1);
    %    X_tr = X_red(:, :, win_n:n);
    %    d_tr = d(1, win_n:n);
    %    if(last_training == 0 || (n - last_training) >= refresh_rate)
    %      disp('Baseline model training --- parallel')
    %      tic()
    %      [centers, rho_base] = learnBaselineModel(K, X_tr, d_tr);
    %      elapsed = toc()
    %      baseline_training_parallel(K, n + 1) = elapsed;
    %    endif
    %    % predict baseline
    %    disp('Baseline model prediction --- parallel')
    %    tic()
    %    [p_0, p_1] = predictBaseline(X_red(:, :, n + 1), centers, rho_base);
    %    elapsed = toc()
    %    baseline_prediction_parallel(K, n + 1) = elapsed;
    %    baseline_correctness_parallel(K, n + 1) = double((p_0 < p_1) == d(n + 1));

    %    if(last_training == 0 || (n - last_training) >= refresh_rate)
    %      disp('Multi-mixture model training --- parallel')
    %      tic()
    %      [mus, Sigmas, rho, pi] = learnExactIndependent(K, X_tr, d_tr, 30);
    %      elapsed = toc()
    %    endif
    %    prob_model_training_parallel(K, n + 1) = elapsed;
    %    disp('Multi-mixture model prediction --- parallel')
    %    tic()
    %    [p_0, p_1] = predictExactIndependent(X_red(:, :, n + 1), mus, Sigmas, rho, pi);
    %    elapsed = toc()
    %    prob_model_prediction_parallel(K, n + 1) = elapsed;
    %    prob_model_correctness_parallel(K, n + 1) = double((p_0 < p_1) == d(n + 1));
    %    prob_model_correctness_parallel(K, 1:n + 1);
    %    p_0
    %    p_1

    %    if(last_training == 0 || (n - last_training) >= refresh_rate && K == max_K)
    %      last_training = n;
    %    endif

        % better save than sorry

    % Update the position of our result pointer.
    test_idx += length(d_test);

    try
      save experimentResultsParallel.mat d min_K max_K S bernoulli_correctness_parallel bernoulli_training_parallel bernoulli_prediction_parallel
      save experimentResultsParallelRelevantServices.mat services dims
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
    %baseline_correctness_parallel
    %baseline_training_parallel
    %baseline_prediction_parallel
    %prob_model_correctness_parallel
    %prob_model_training_parallel
    %prob_model_prediction_parallel
    %svm_correctness_parallel
    %svm_training_parallel
    %svm_prediction_parallel
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

