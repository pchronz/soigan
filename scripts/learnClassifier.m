function CC = learnClassifier(D_train, classlabel_train)
  % TODO try the following scheme:
  % 1 undersampling of the majority class to obtain a balanced set
  % 2 train multiple (how many?) SVMs to obtain various experts
  % 3 create a mixture of experts using the SVMs
  % TODO try using the RVM
  % TODO optimize the parameters using simulated annealing or another nonlinear optimization technique
  % this should incorporate a higher penalty to represent the gravity of false successes

  % XXX thought about trying PCA, but this really does not help all that much here.
  % PCA is not really helpful for classification because it does not take take the 
  % labels into account and thus may be a hindrance to the classification
  % Fisher's linear discriminant is the way to go since it provides a projection which
  % does take the labels into account
  num_runs=2;
  hypers=zeros(num_runs,3);
  MODE.TYPE='rbf';
  for it=1:num_runs
    it
    % cross-validation as another inner loop
    MODE.hyperparameter.c_value=rand(1)*250;
    MODE.hyperparameter.gamma=rand(1)/10000;
    hypers(it,2)=MODE.hyperparameter.c_value;
    hypers(it,3)=MODE.hyperparameter.gamma;
    num_bins=5;
    errors_cross=zeros(1,num_bins);
    for s=1:num_bins
      % split the training set into a training and a test set
      [D_tr, classlabel_tr, D_te, classlabel_te] = split_data(D_train, classlabel_train, 1/num_bins, s, 1);

      % train
      CC=train_sc(D_tr,classlabel_tr,MODE);

      % test with test set within cross-validation
      R=test_sc(CC,D_te);
      % increase the penalty for the false failures
      delta=classlabel_te'-R.classlabel;
      % experiment with and without penalty
      delta(delta>0)=delta(delta>0);
      errors_cross(s)=sum(abs(delta));
    endfor
    hypers(it,1)=mean(errors_cross);
    hypers(it,1)
  endfor

  % choose the best model found through cross validation
  [e,i]=sort(hypers(:,1));
  MODE.hyperparameter.c_value=hypers(i(1),2);
  MODE.hyperparameter.gamma=hypers(i(1),3);
  CC=train_sc(D_train,classlabel_train,MODE);
endfunction

