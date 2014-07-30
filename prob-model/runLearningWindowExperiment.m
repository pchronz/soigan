function runLearningWindowExperiment()
  more off

  pkg load statistics
  pkg load nan

  clear

  min_N = 1;
  max_N = 500;

  refresh_rate = 10;

  global deter = false;
  X = 0;
  d = 0;
  if(deter)
    disp('Using previously used data set.')
    load('detdata.mat');
  else
    disp('Preparing data set from scratch.')
    % N, I, R, D, K
    %[X, d] = createSyntheticData(max_N, 5, 3, 3, 3);
    [X, d] = loadEwsData(max_N);
    %[X, d] = loadGoeGridFullData(0, max_N);
    save detdata.mat X d
  endif

  % Ensure that we have a value for each class in the data set.
  first_neg = min([1:max_N](d == 0));
  first_pos = min([1:max_N](d == 1));
  min_N = max([first_neg, first_pos, min_N]);

  [D, I, N] = size(X);

  disp('Running full SVM training')
  [svm_training, svm_pred, svm_corr] = runSvmExperiment(min_N, X, d, refresh_rate);

  disp('Computing quality for full SVM experiment')
  [prec, rec, F] = computeQuality(svm_corr, d, min_N);

  disp('Computing window length')
  win_len = min_N + computeLearningWindow(F(min_N + 1:end), 0.05, 30)

  % Re-run serial SVM experiment with the new window length to verify that the quality is fine.
  disp('Running windowed SVM training')
  [svm_training_win, svm_pred_win, svm_corr_win] = runSvmExperiment(min_N, X, d, refresh_rate, win_len);

  disp('Computing quality for windowed SVM experiment')
  [prec_win, rec_win, F_win] = computeQuality(svm_corr_win, d, min_N);

  % Train the full probabilistic model once for the sample size of n, without windowing.
  n = min(500, N - 1);
  K = 2;
  disp(['Running full prob model training for n = ' num2str(n)])
  [prob_train_full, prob_pred_full, prob_corr_full] = runProbModelExperiment(K, X, d, min_N, refresh_rate);
  disp('Computing quality for full probabilistic experiment')
  [prec_prob_full, rec_prob_full, F_prob_full] = computeQuality([prob_corr_full], [d(n)], 1);
  % Train the full probabilistic model for a sample size of n and a window of win_len.
  disp(['Running full prob model training for n = ' num2str(n) ' and a window with win_len = ' num2str(win_len)])
  [prob_train_win, prob_pred_win, prob_corr_win] = runProbModelExperiment(K, X, d, min_N, refresh_rate, win_len);
  disp('Computing quality for windowed probabilistic experiment')
  [prec_prob_win, rec_prob_win, F_prob_win] = computeQuality([prob_corr_full], [d(n)], 1);
  % Compare the performance of the windowed and non-windowed run: duration of training, duration of prediction, and F-measure.

  % Plot
  subplot(3, 1, 1)
  plot([F; F_win; F_prob_full; F_prob_win]')
  F_prob_win
  F_prob_full
  subplot(3, 1, 2)
  plot([prob_train_full; prob_train_win]')
  subplot(3, 1, 3)
  plot([prob_pred_full; prob_pred_win]')
endfunction

function [svm_training_serial, svm_prediction_serial, svm_correctness_serial] =  runSvmExperiment(min_N, X, d, refresh_rate, win_len = Inf)
  [D, I, N] = size(X);
  svm_training_serial = zeros(1, N);
  svm_prediction_serial = zeros(1, N);
  svm_correctness_serial = zeros(1, N);

  last_training = 0;
  % SVM
  for n = min_N:N - 1
    n

    win_n = max(1, n - win_len + 1);

    X_tr = X(:, :, win_n:n);
    d_tr = d(1, win_n:n);
    if(last_training == 0 || (n - last_training) >= refresh_rate)
      disp('SVM training')
      % learn SVM
      tic()
      MODE.TYPE='rbf';
      MODE.hyperparameter.c_value=rand(1)*250;
      MODE.hyperparameter.gamma=rand(1)/10000;
      if(sum(d_tr) > 0 && sum(!d_tr) > 0)
        [X_tr, d_tr] = balanceData(X_tr, d_tr);
      endif
      [D, I, N] = size(X_tr);
      CC = train_sc(reshape(X_tr, [D*I, N])', (d_tr + 1)', MODE);
      % XXX why does the SVM not work below a certain number of training vectors?
      % ==> cause it ain't got more than one class observed!
      elapsed = toc();
      svm_training_serial(n) = elapsed;
      last_training = n;
    endif
    if(CC.model.totalSV > 0)
      disp('SVM prediction')
      % predict SVM
      tic();
      [D, I, N_te] = size(X(:, :, n + 1));
      hits_svm = test_sc(CC, reshape(X(:, :, n + 1), [D*I, N_te])');
      hits_svm = hits_svm.classlabel - 1;
      hits_svm = sum(hits_svm == d(1, n + 1));
      elapsed = toc();
      svm_prediction_serial(n) = elapsed;
      assert(hits_svm == 0 || hits_svm == 1);
      svm_correctness_serial(n) = hits_svm;
    else
      % disp('SVM model failed at...')
      % n
      svm_training_serial(n) = -1;
      svm_correctness_serial(n) = -1;
      svm_prediction_serial(n) = -1;
    endif
  endfor

  svm_training_serial(n) = elapsed;
  svm_prediction_serial(n) = elapsed;
  svm_correctness_serial(n) = hits_svm;
endfunction

function [precision, recall, F_meas] = computeQuality(corr, d, min_N)
  N = length(d);
  % Compute the F-measure
  precision = zeros(1, N);
  recall = zeros(1, N);
  F_meas = zeros(1, N);

  for n = min_N + 1:N
    tp = sum(corr(min_N:n) .* d(min_N:n));
    fp = sum((!corr(min_N:n)) .* (!d(min_N:n)));
    % Precision
    if(tp + fp != 0)
      precision(n) = tp/(tp + fp);
    endif
    % Recall
    if(sum(d(min_N:n)) != 0)
      recall(n) = tp/sum(d(min_N:n));
    endif
    % F-measure
    if(precision(n) + recall(n) != 0)
      F_meas(n) = 2*precision(n)*recall(n)/(precision(n) + recall(n));
    endif
  endfor
endfunction

function [prob_train, prob_pred, prob_corr] = runProbModelExperiment(K, X, d, min_N, refresh_rate, win_len = Inf)
  [D, I, N] = size(X);
  prob_train = zeros(1, N);
  prob_pred = zeros(1, N);
  prob_corr = zeros(1, N);
  last_training = 0;
  % Dimensions to use for the full probabilistic model.
  dims = linspace(1, I, I);
  tic()
  disp('Running dimensionality reduction')
  dims = crossReduceDimensions(X, d);
  toc()
  last_training = 0;
  for n = min_N:N - 1
    win_n = max(1, n - win_len + 1);
    X_tr = X(:, :, 1:n);
    d_tr = d(1, 1:n);

    % Probabilistic model
    if(last_training == 0 || (n - last_training) >= refresh_rate)
      disp('Multi-mixture model training --- serial')
      tic()
      [mus, Sigmas, rho, pi] = learnExactIndependent(K, X_tr(:, dims, :), d_tr, 10);
      elapsed = toc()
    endif
    prob_train(n) = elapsed;
    disp('Multi-mixture model prediction --- serial')
    tic()
    [p_0, p_1] = predictExactIndependent(X(:, dims, n + 1), mus, Sigmas, rho, pi);
    elapsed = toc()
    prob_pred(n) = elapsed;
    prob_corr(n) = double((p_0 < p_1) == d(n + 1));
    p_0
    p_1

    if(last_training == 0 || (n - last_training) >= refresh_rate)
      last_training = n;
    endif
    
    n
  endfor
endfunction

