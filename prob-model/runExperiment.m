more off

pkg load statistics
pkg load nan

% TODO sampling of exact posterior
% TODO maximization based on sampling
% TODO scenario ranking
% TODO dimension ranking
% TODO variational inference
% TODO detect singularities

% create the data first
% D x I x N
N = 500;
% mixture components
I = 5;
D = 3;
K = 2;
X_1 = mvnrnd(0.7 * ones(1, D), 0.01 * eye(D) + 0 * rotdim(eye(D), 1), I*N)';
%X_1 = mvnrnd(0.7 * ones(1, D), 0.01 * eye(D) + 0 * rotdim(eye(D), 1), I*N*1/4)';
%X_2 = mvnrnd(0.3 * ones(1, D), 0.05 * eye(D) + 0 * rotdim(eye(D), 1), I*N*2/4)';
%X_3 = mvnrnd(0.1 * ones(1, D), 0.001 * eye(D) + 0 * rotdim(eye(D), 1), I*N*1/4)';
perm_vec = randperm(I*N);
%X = reshape([X_1, X_2, X_3](:, perm_vec), [D, N, I]);
X = reshape(X_1(:, perm_vec), [D, N, I]);
% mark the clusters after permutation
clusters = reshape((X_1(1, :) > 0.7)(perm_vec), [N, I]);
% d := 1, if at least two clusters are ones
d = (sum(clusters, 2) >= 2)';
X = rotdim(X, 1, [2, 3]);

% the experimental parameters
Delay = [0:0];
It = 20;
max_K = 5;
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

% SERIAL EXPERIMENT
% [X, d] = loadEwsData();
[X, d] = loadGoeGridData();
% [X, d] = loadHEPhyData();
% [X, d] = loadGoeGridFullData(0);
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

% SVM
for n = 1:N - 1
  disp('SVM training and prediction')
  n

  X_tr = X(:, :, 1:n);
  d_tr = d(1, 1:n);
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
  if(CC.model.totalSV > 0)
    svm_training_serial(n) = elapsed;
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
disp('SVM failures for n = ')
[D, I, N] = size(X);
[1:N](svm_correctness_serial == -1)

% going one step ahead is fine as long as the time required for prediction and training
% is less than the current time plus the time for which we wish to predict. Actually there
% we also need to count in a slack that is the time until which the observed global value
% (via Cern) will be available and the time that we need to react and fix an error (MTTR).
% baseline & prob model
for n = max_K:N - 1
  for K = 2:max_K
    disp('n -- serial')
    disp([num2str(n), '/', num2str(N - 1), ' = ', num2str(n/N), '%'])
    disp('K -- serial')
    disp([num2str(K), '/', num2str(max_K), ' = ', num2str(K/max_K), '%'])
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
    X_tr = X(:, :, 1:n);
    d_tr = d(1, 1:n);
    disp('Baseline model training --- serial')
    tic()
    [centers, rho_base] = learnBaselineModel(K, X_tr, d_tr);
    elapsed = toc()
    baseline_training_serial(K, n + 1) = elapsed;
    % predict baseline
    disp('Baseline model prediction --- serial')
    tic()
    [p_0, p_1] = predictBaseline(X(:, :, n + 1), centers, rho_base);
    elapsed = toc()
    baseline_prediction_serial(K, n + 1) = elapsed;
    baseline_correctness_serial(K, n + 1) = double((p_0 < p_1) == d(n + 1));

    % disp('Multi-mixture model training --- serial')
    % tic()
    % [mus, Sigmas, rho, pi] = learnExactIndependent(K, X_tr, d_tr, 10);
    % elapsed = toc()
    % prob_model_training_serial(K, n + 1) = elapsed;
    % disp('Multi-mixture model prediction --- serial')
    % tic()
    % [p_0, p_1] = predictExactIndependent(X(:, :, n + 1), mus, Sigmas, rho, pi);
    % elapsed = toc()
    % prob_model_prediction_serial(K, n + 1) = elapsed;
    % prob_model_correctness_serial(K, n + 1) = double((p_0 < p_1) == d(n + 1));

    % better save than sorry
    save experimentResultsSerial.mat d max_K baseline_correctness_serial baseline_training_serial baseline_prediction_serial prob_model_correctness_serial prob_model_training_serial prob_model_prediction_serial svm_correctness_serial svm_training_serial svm_prediction_serial
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

% PARALLEL EXPERIMENT
for delay = Delay
  disp('Delay...')
  delay

  for it = 1:It
    disp('Experiment iteration...')
    it

    % [X, d] = loadEwsData();
    [X, d] = loadGoeGridData();
    % [X, d] = loadHEPhyData();
    % [X, d] = loadGoeGridFullData(delay);
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
    for K = 2:max_K
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


