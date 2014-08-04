more off
clear

pkg load statistics
pkg load nan
pkg load parallel

D = 5;
I = 7;
K = 2;
N = 500;

mus = rand(D, K, I);
Sigmas = eye(D)(:, :, ones(1, K), ones(1, I));
pi = 1/K*ones(K, I);
rho = 0.5 * ones(K^I, 1);
X = rand(D, I, N);
d = binornd(ones(1, N), 0.5);

% Before optimization (D, I, K, N) = (5, 5, 2, 200) --> ~= 13s.
% Moving memory allocation out of the loops cut it down to ~= 6.5s.
% Vectorization of mvnpdf got it down to ~= 5.3s.
profile on
tic()
p_Z_vec = computePosteriorVectorized(mus, Sigmas, pi, rho, X, d, K);
toc()
profile off
p = profile('info');
profshow(p)

profile on
tic()
p_Z_slo = computePosteriorSlow(mus, Sigmas, pi, rho, X, d, K);
toc()
profile off
p = profile('info');
profshow(p)

mean(mean(abs(p_Z_slo - p_Z_vec)))

