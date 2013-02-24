more off

pkg load statistics

% TODO sampling
% TODO variational inference
% TODO Bayesian

% create the data first
% D x I x N
N = 50;
% mixture components
I = 3;
D = 3;
X_1 = mvnrnd(0.7 * ones(1, D), 0.01 * eye(D) + 0 * rotdim(eye(1), 1), N/2)';
X_2 = mvnrnd(0.3 * ones(1, D), 0.01 * eye(D) + 0 * rotdim(eye(1), 1), N/2)';
X = zeros(1, I, N);
X = [X_1, X_2](:, :, ones(1, I));
X = rotdim(X, 1, [2, 3]);
% the global system state observations for now: 1 if mixture k=1, 0 if mixture k=2
% TODO train specific combinations of mixtures to be 1 or 0
d = [zeros(1, N/2), ones(1, N/2)];

[mus, Sigmas, rho, pi] = learnFailurePredictor(2, X, d, 50);

% evalute the Akaike information criterion
aic = computeAic(mus, Sigmas, rho, pi, X, d);

rho
mus
Sigmas
pi
aic

X_next = 0.3 * ones(D, I);
d_next = 0;
[p_0, p_1] = predictFailure(X_next, mus, Sigmas, rho, pi);
p_0
p_1





