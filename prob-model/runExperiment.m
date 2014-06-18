more off

pkg load statistics
pkg load nan

% TODO sampling of exact posterior
% TODO maximization based on sampling
% TODO scenario ranking
% TODO dimension ranking
% TODO variational inference
% TODO detect singularities

% the experimental parameters
Delay = [0:0];
It = 20;
min_K = 2;
max_K = 5;

% N, I, R, D, K
[X, d] = createSyntheticData(25, 5, 3, 3, 3);
% [X, d] = loadEwsData();
% [X, d] = loadGoeGridData();
% [X, d] = loadHEPhyData();
% [X, d] = loadGoeGridFullData(0);

disp(sum(d == 0))
disp(sum(d == 1))

[baseline_correctness_serial, baseline_training_serial, baseline_prediction_serial, prob_model_correctness_serial, prob_model_training_serial, prob_model_prediction_serial, svm_correctness_serial, svm_training_serial, svm_prediction_serial] = runSerialExperiment(X, d, min_K, max_K);
%[baseline_accuracy, prob_accuracy, svm_accuracy, baseline_learning, prob_learning, svm_learning, baseline_prediction, prob_prediction, svm_prediction] = runParallelExperiment(X, d, Delay, It, min_K, max_K);

