more off
clear

pkg load statistics
pkg load nan
pkg load parallel

D = 5;
I = 3;
K = 2;
N = 250;

mus = rand(D, K, I);
Sigmas = eye(D)(:, :, ones(1, K), ones(1, I));
% Randomize and bias the values.
pi = rand(K, I);
pi(ceil(K/2) + 1:end, :) = pi(ceil(K/2) + 1:end, :) * 100;
pi = 1./sum(pi).*pi;
%pi = 1/K*rand(K, I);
rho = 0.5 * ones(K^I, 1);
X = rand(D, I, N);
d = binornd(ones(1, N), 0.5);

tic()
p_Z_appr = computePosteriorApproximate(mus, Sigmas, pi, rho, X, d, K);
toc()

tic()
p_Z_slo = computePosteriorSlow(mus, Sigmas, pi, rho, X, d, K);
toc()

tic()
p_Z_vec = computePosteriorVectorized(mus, Sigmas, pi, rho, X, d, K);
toc()

mean(mean(abs(p_Z_slo - p_Z_vec)))

