more off

pkg load statistics
pkg load nan
pkg load parallel

clear all

% TODO RCA
% TODO variational inference

% the experimental parameters
Delay = [0:0];
It = 15;
min_K = 2;
max_K = 2;
min_N = 1;
max_N = 500;
refresh_rate = 100;
win_len = 1000;
cross_S = 8;

global deter = false;
global para = true;
global crossred = true;

X = 0;
d = 0;
if(deter)
  disp('Using previously used data set.')
  load('detdata.mat');
else
  disp('Preparing data set from scratch.')
  % N, I, R, D, K
  %[X, d] = createSyntheticData(max_N, 5, 3, 3, 3);
  %[X, d] = loadEwsData(max_N);
  [X, d] = loadGoeGridFullData(0, max_N);
  save detdata.mat X d
endif

disp(sum(d == 0))
disp(sum(d == 1))

%[baseline_correctness_serial, baseline_training_serial, baseline_prediction_serial, prob_model_correctness_serial, prob_model_training_serial, prob_model_prediction_serial, svm_correctness_serial, svm_training_serial, svm_prediction_serial] = runSerialExperiment(X, d, min_K, max_K, min_N, refresh_rate, win_len, cross_S);
runParallelExperiment(X, d, min_K, max_K, 8);

