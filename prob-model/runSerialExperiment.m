function [baseline_correctness_serial, baseline_training_serial, baseline_prediction_serial, prob_model_correctness_serial, prob_model_training_serial, prob_model_prediction_serial, svm_correctness_serial, svm_training_serial, svm_prediction_serial] = runSerialExperiment(X, d, min_K, max_K)
  % SERIAL EXPERIMENT
  [D, I, N] = size(X);
  % result containers
  baseline_correctness_serial = zeros(max_K, N);
  baseline_training_serial = zeros(max_K, N);
  baseline_prediction_serial = zeros(max_K, N);
  prob_model_correctness_serial = zeros(max_K, N);
  prob_model_training_serial = zeros(max_K, N);
  prob_model_prediction_serial = zeros(max_K, N);
  svm_correctness_serial = zeros(1, N);
  svm_training_serial = zeros(1, N);
  svm_prediction_serial = zeros(1, N);
  bernoulli_correctness_serial = zeros(1, N);
  bernoulli_training_serial = zeros(1, N);
  bernoulli_prediction_serial = zeros(1, N);
  
%  % Bernoulli (max likelihood)
%  for n = 2:N - 1
%      disp('Bernoulli model training --- serial')
%      tic()
%      % TODO Try out setting rho to 0.5, which someone might do naively.
%      rho = sum(d(1, 1:n))/N
%      elapsed = toc()
%      bernoulli_training_serial(1, n + 1) = elapsed;
%      disp('Bernoulli model prediction --- serial')
%      tic()
%      % Choose a random value and compare to rho.
%      bernoulli_correctness_serial(1, n + 1) = double(rand() > rho == d(n + 1));
%      elapsed = toc()
%      bernoulli_prediction_serial(1, n + 1) = elapsed;
%  endfor
%
%  % SVM
%  for n = 1:N - 1
%    disp('SVM training and prediction')
%    n
%
%    X_tr = X(:, :, 1:n);
%    d_tr = d(1, 1:n);
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
%    if(CC.model.totalSV > 0)
%      svm_training_serial(n) = elapsed;
%      % predict SVM
%      tic();
%      [D, I, N_te] = size(X(:, :, n + 1));
%      hits_svm = test_sc(CC, reshape(X(:, :, n + 1), [D*I, N_te])');
%      hits_svm = hits_svm.classlabel - 1;
%      hits_svm = sum(hits_svm == d(1, n + 1));
%      elapsed = toc();
%      svm_prediction_serial(n) = elapsed;
%      assert(hits_svm == 0 || hits_svm == 1);
%      svm_correctness_serial(n) = hits_svm;
%    else
%      % disp('SVM model failed at...')
%      % n
%      svm_training_serial(n) = -1;
%      svm_correctness_serial(n) = -1;
%      svm_prediction_serial(n) = -1;
%    endif
%  endfor
%  disp('SVM failures for n = ')
%  [D, I, N] = size(X);
%  [1:N](svm_correctness_serial == -1)

  % going one step ahead is fine as long as the time required for prediction and training
  % is less than the current time plus the time for which we wish to predict. Actually there
  % we also need to count in a slack that is the time until which the observed global value
  % (via Cern) will be available and the time that we need to react and fix an error (MTTR).
  % baseline & prob model
  %for n = max_K:N - 1
  for n = 123:N - 1
    for K = min_K:max_K
      disp('n -- serial')
      disp([num2str(n), '/', num2str(N - 1), ' = ', num2str(n/N)*100, '%'])
      disp('K -- serial')
      disp([num2str(K), '/', num2str(max_K), ' = ', num2str(K/max_K)*100, '%'])
      % XXX review ETA, since K and n have been swapped
      if(n >= max_K + 25)
        slope = 19/(mean(baseline_training_serial(K, n - 6:n)) - mean(baseline_training_serial(K, n - 25:n - 19)));
        remaining_iterations = N - n + (N - max_K)*(max_K - K);
        % total[s], d, h, m, s
        ETA = zeros(1, 5);
        ETA(1) = slope * remaining_iterations;
        ETA(2) = floor(ETA(1)/(60*60*24));
        ETA(3) = floor(mod(ETA(1), 60*60*24)/(60*60));
        ETA(4) = floor(mod(ETA(1), 60*60)/(60));
        ETA(5) = floor(mod(ETA(1), 60));
        disp(['ETA: ', num2str(ETA(2)), 'd ', num2str(ETA(3)), 'h ', num2str(ETA(4)), 'm ', num2str(ETA(5)), 's'])
      endif
      % TODO What happens if we balance the data set first as for the SVM?
      X_tr = X(:, :, 1:n);
      d_tr = d(1, 1:n);
      %disp('Baseline model training --- serial')
      %tic()
      %[centers, rho_base] = learnBaselineModel(K, X_tr, d_tr);
      %elapsed = toc()
      %baseline_training_serial(K, n + 1) = elapsed;
      %% predict baseline
      %disp('Baseline model prediction --- serial')
      %tic()
      %[p_0, p_1] = predictBaseline(X(:, :, n + 1), centers, rho_base);
      %elapsed = toc()
      %baseline_prediction_serial(K, n + 1) = elapsed;
      %baseline_correctness_serial(K, n + 1) = double((p_0 < p_1) == d(n + 1));

      disp('Multi-mixture model training --- serial')
      tic()
      [mus, Sigmas, rho, pi] = learnExactIndependent(K, X_tr, d_tr, 10);
      elapsed = toc()
      prob_model_training_serial(K, n + 1) = elapsed;
      disp('Multi-mixture model prediction --- serial')
      tic()
      [p_0, p_1] = predictExactIndependent(X(:, :, n + 1), mus, Sigmas, rho, pi);
      elapsed = toc()
      prob_model_prediction_serial(K, n + 1) = elapsed;
      prob_model_correctness_serial(K, n + 1) = double((p_0 < p_1) == d(n + 1));
      prob_model_correctness_serial(K, 1:n + 1)
      p_0
      p_1

      % better save than sorry
      save experimentResultsSerial.mat d max_K baseline_correctness_serial baseline_training_serial baseline_prediction_serial prob_model_correctness_serial prob_model_training_serial prob_model_prediction_serial svm_correctness_serial svm_training_serial svm_prediction_serial bernoulli_correctness_serial bernoulli_training_serial bernoulli_prediction_serial
      disp('The serial results have been saved')

    endfor
  endfor

  baseline_correctness_serial
  baseline_training_serial
  baseline_prediction_serial
  prob_model_correctness_serial
  prob_model_training_serial
  prob_model_prediction_serial
  svm_correctness_serial
  svm_training_serial
  svm_prediction_serial
  bernoulli_correctness_serial
  bernoulli_training_serial
  bernoulli_prediction_serial
endfunction
