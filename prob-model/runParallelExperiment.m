function [baseline_accuracy, prob_accuracy, svm_accuracy, baseline_learning, prob_learning, svm_learning, baseline_prediction, prob_prediction, svm_prediction] = runParallelExperiment(X, d, Delay, It, min_K, max_K)

  % PARALLEL EXPERIMENT

  % the experimental results
  % accuracies
  baseline_accuracy = zeros(length(Delay), It, max_K);
  prob_accuracy = zeros(length(Delay), It, max_K);
  svm_accuracy = zeros(length(Delay), It, max_K);
  % learning times
  baseline_learning = zeros(length(Delay), It, max_K);
  prob_learning = zeros(length(Delay), It, max_K);
  svm_learning = zeros(length(Delay), It, max_K);
  % training times
  baseline_prediction = zeros(length(Delay), It, max_K);
  prob_prediction = zeros(length(Delay), It, max_K);
  svm_prediction = zeros(length(Delay), It, max_K);

  % save the original data
  X_orig = X;
  d_orig = d;

  for delay = Delay
    disp('Delay...')
    delay

    for it = 1:It
      disp('Experiment iteration...')
      it

      X = X_orig;
      d = d_orig;
      [D, I, N] = size(X);
      % XXX uncomment to run on a smaller dataset for development
      % N = 30;
      % X = X(:, :, 1:N);
      % d = d(1:N);
      % assert(size(X) == [D, I, N])
      % assert(size(d) == [1, N])


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
        idx = idx < 8;
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
      [X_test, d_test] = balanceData(X_test, d_test);
      X = X(:, :, [idx_0_train, idx_1_train]);
      d = d([idx_0_train, idx_1_train]);
      [X, d] = balanceData(X, d);

      % run multiple values of K on the same data
      for K = min_K:max_K
        disp('Baseline model training --- parallel')
        tic()
        [centers, rho_base] = learnBaselineModel(K, X, d);
        elapsed = toc()
        baseline_learning(delay - min(Delay) + 1, it, K) = elapsed;

        disp('SVM training --- parallel')
        tic()
        MODE.TYPE='rbf';
        MODE.hyperparameter.c_value=rand(1)*250;
        MODE.hyperparameter.gamma=rand(1)/10000;
        [D, I, N_tr] = size(X);
        CC = train_sc(reshape(X, [D*I, N_tr])', (d + 1)', MODE);
        elapsed = toc()
        svm_learning(delay - min(Delay) + 1, it, K) = elapsed;

        % disp('Multi-mixture model training')
        % tic()
        % [mus, Sigmas, rho, pi] = learnExactIndependent(K, X, d, 10);
        % elapsed = toc()
        % prob_learning(delay - min(Delay) + 1, it, K) = elapsed;

        % evalute the Akaike information criterion
        %aic = computeAicIndependent(mus, Sigmas, rho, pi, X, d);
        %
        %rho
        %mus
        %Sigmas
        %pi
        %aic

        N_test = size(X_test, 3);
        hits = 0;
        hits_baseline = 0;

        disp('Predictions prob model --- parallel')
        tic()
        for n = 1:N_test
          %  [p_0, p_1] = predictExactIndependent(X_test(:, :, n), mus, Sigmas, rho, pi);
          %  hits = hits + double((p_0 < p_1) == d_test(n));
        endfor
        elapsed = toc()
        prob_prediction(delay - min(Delay) + 1, it, K) = elapsed;
        disp('Predictions baseline model')
        tic()
        for n = 1:N_test
          [p_0, p_1] = predictBaseline(X_test(:, :, n), centers, rho_base);
          hits_baseline = hits_baseline + double((p_0 < p_1) == d_test(n));
        endfor
        elapsed = toc()
        baseline_prediction(delay - min(Delay) + 1, it, K) = elapsed;
        disp('Predictions SVM --- parallel')
        tic()
        [D, I, N_te] = size(X_test);
        hits_svm = test_sc(CC, reshape(X_test, [D*I, N_te])');
        hits_svm = hits_svm.classlabel - 1;
        hits_svm = sum(hits_svm == d_test)
        elapsed = toc()
        svm_prediction(delay - min(Delay) + 1, it, K) = elapsed;

        disp('prob model accuracy --- parallel');
        hits / N_test
        prob_accuracy(delay - min(Delay) + 1, it, K) = hits / N_test;
        disp('baseline model accuracy --- parallel');
        hits_baseline / N_test
        baseline_accuracy(delay - min(Delay) + 1, it, K) = hits_baseline / N_test;
        disp('SVM accuracy --- parallel');
        hits_svm / N_test
        svm_accuracy(delay - min(Delay) + 1, it, K) = hits_svm / N_test;
      endfor
    endfor

    delay
    disp('prob model mean and standard deviation')
    mean(prob_accuracy(delay - min(Delay) + 1, :, :), 2)
    std(prob_accuracy(delay - min(Delay) + 1, :, :), 0, 2)

    disp('baseline model mean and standard deviation')
    mean(baseline_accuracy(delay - min(Delay) + 1, :, :), 2)
    std(baseline_accuracy(delay - min(Delay) + 1, :, :), 0, 2)

    % better save than sorry
    save experimentResultsParallel.mat It max_K Delay baseline_accuracy baseline_learning baseline_prediction prob_accuracy prob_learning prob_prediction svm_accuracy svm_learning svm_prediction
    disp('The parallel results have been saved')
  endfor

  baseline_accuracy

  svm_accuracy
endfunction

