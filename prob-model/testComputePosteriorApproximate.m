more off
clear

pkg load statistics
pkg load nan
pkg load parallel

D = 10;
I = 6;
K = 3;
N = 50;

mus = rand(D, K, I);
Sigmas = eye(D)(:, :, ones(1, K), ones(1, I));
% Randomize and bias the values for pi.
pi = rand(K, I);
pi(ceil(K/2) + 1:end, :) = pi(ceil(K/2) + 1:end, :) * 100;
pi = 1./sum(pi).*pi;
%pi = 1/K*rand(K, I);
% Randomize and bias the values for rho.
rho = rand(K^I, 1);
rho(1:0.5*(K^I), 1) = 0.01*rand(0.5*K^I, 1);
X = rand(D, I, N);
%rho = rand(K^I, 1);
d = binornd(ones(1, N), 0.5);

tic()
p_Z_appr_vec = computePosteriorApproximateVectorized(mus, Sigmas, pi, rho, X, d, K);
toc()

tic()
p_Z_appr = computePosteriorApproximate(mus, Sigmas, pi, rho, X, d, K);
toc()

%tic()
%p_Z_slo = computePosteriorSlow(mus, Sigmas, pi, rho, X, d, K);
%toc()
%
%tic()
%p_Z_vec = computePosteriorVectorized(mus, Sigmas, pi, rho, X, d, K);
%toc()

%p_Z_slo
%p_Z_appr
mean(mean(abs(p_Z_appr - p_Z_appr_vec)))
mean(mean(abs(p_Z_appr - p_Z_slo)))

