pkg load lds
pkg load nan
more off

disp('reading in the data');
% only read in the file if the data variable does not already exist
if(exist('data') == 0)
  % load the csv: ',' as seperator, starting with the second line
  data=dlmread('../data/creamce_with_sam_1.csv',',',1,0);
end

tic()
% seperate the inputs from the results
% and kick out the timestamps
% TODO retaining only the numerical columns for now
D=data(:,2:13);
% removing all target dimensions that do not have different values assigned for the observations
idx=var(data(:, 14:end))==0;
data(:, idx) = [];
classlabels = data(:, 14:end);
classlabels = data(:, 17);
toc()

% automatically remove the columns which have zero variance
idx = var(D) == 0;
D(:, idx) = [];
assert(sum(var(D) == 0) == 0)
D_orig = D;

for cl = 1:length(classlabels(1, :))
  figure("visible", "off")

  classlabel = classlabels(:, cl);
  % XXX merge OK and WARNING
  idx = (classlabel == 4);
  classlabel(idx) = 1;

  % MISSING
  idx = (classlabel != 3);
  D = D_orig(idx, :);
  classlabel = classlabel(idx);

  classlabel_orig = classlabel;

  min_w = 0;
  max_w = 0;
  errors_mean = zeros(max_w - min_w + 1, 4);
  errors_var = zeros(max_w - min_w + 1, 4);
  for w = min_w : max_w;
    w

    %disp('Starting to train the LDS')
    %tic()
    % XXX filtering does not seem to work --> after 2 iterations the learner runs into a singularity
    %% learn the LDS and apply the resulting Kalman filter for the preprocessing phase as smoother
    %D = preprocessBySplitting(D_orig, classlabel_orig, w);
    %dim_z = 3;
    %[lds, muhats, likelihoods] = learnLDS(D, 5, dim_z, 0.1);
    %M = length(D(1, 1, :));
    %D_filtered = zeros(M * (w + 1), dim_z);
    %for m = 1:M
    %  [Z_mu, Z_cov] = filterSequence(lds, D(:, :, m));
    %  D_filtered(Z_mu')
    %endfor
    %toc()

    % [D, classlabel] = preprocessByAggregation(D_orig, classlabel_orig);
    [D, classlabel] = preprocessByConcatenation(D_orig, classlabel_orig, w);
    disp('#########################################################')
    disp('Number of class speciments in the training and test set: ');
    disp(sum(classlabel == 1))
    disp(sum(classlabel == 2))

    % TODO XXX why are the results so bad if I switch preprocessing and normalization?
    % again prune dimensions with zero-variance this is needed since new 'uninformative dimensions might have been introduced due to changing the format of the data
    D(:, var(D) == 0) = [];
    % normalize the data
    m=mean(D);
    sigma=var(D);
    D = bsxfun(@minus, D, m);
    D = bsxfun(@rdivide, D, sigma);

    num_experiments = 20;
    errors_exp = zeros(num_experiments, 3);
    for experiment = 1 : num_experiments 
      experiment

      idx=randperm(length(classlabel));
      classlabel=classlabel(idx);
      D=D(idx,:);

      % 20<->80 split (test<->train) is a widely used convention [GMNS2009, CB2008]
      [D_train, classlabel_train, D_test, classlabel_test] = split_data(D, classlabel, 0.2, 1, 0);

      % learn it
      disp('starting the training');
      tic();
      CC = learnClassifier(D_train, classlabel_train);
      toc();

      % validate it
      [e_total, e_1, e_2] = validateClassifier(CC, D_test, classlabel_test);
      errors_exp(experiment, 1) = e_total;
      errors_exp(experiment, 2) = e_1;
      errors_exp(experiment, 3) = e_2;
    endfor
    errors_mean(w - min_w + 1, 1) = w;
    errors_mean(w - min_w + 1, 2) = mean(errors_exp(:, 1));
    errors_mean(w - min_w + 1, 3) = mean(errors_exp(:, 2));
    errors_mean(w - min_w + 1, 4) = mean(errors_exp(:, 3));
    errors_var(w - min_w + 1, 1) = w;
    errors_var(w - min_w + 1, 2) = var(errors_exp(:, 1));
    errors_var(w - min_w + 1, 3) = var(errors_exp(:, 2));
    errors_var(w - min_w + 1, 4) = var(errors_exp(:, 3));

    subplot(3, 1, 1)
    plot(errors_mean(:, 1), errors_mean(:, 2))
    subplot(3, 1, 2)
    plot(errors_mean(:, 1), errors_mean(:, 3))
    subplot(3, 1, 3)
    plot(errors_mean(:, 1), errors_mean(:, 4))
    filename = [int2str(cl), "-", int2str(w), ".pdf"];
    print(filename, "-dpdf")
  endfor
end



