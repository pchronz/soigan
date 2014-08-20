more off
clear

pkg load statistics
pkg load nan
pkg load parallel

D = 5;
I = 5;
K = 3;
N = 100;

mus = rand(D, K, I);
Sigmas = eye(D)(:, :, ones(1, K), ones(1, I));
pi = 1/K*ones(K, I);
rho = 0.5 * ones(K^I, 1);
X = rand(D, I, N);
d = binornd(ones(1, N), 0.5);

% Before optimization (D, I, K, N) = (5, 5, 2, 200) --> ~= 13s.
% Moving memory allocation out of the loops cut it down to ~= 6.5s.
% Vectorization of mvnpdf got it down to ~= 5.3s.
tic()
p_Z_vec = computePosteriorVectorized(mus, Sigmas, pi, rho, X, d, K);
toc()

tic()
[mus, pi] = maxMuPi(p_Z_vec, X, K);
toc()

tic()
[mus_vec, pi_vec] = maxMuPiVectorized(p_Z_vec, X, K);
toc()

mean(mean(mean(mus - mus_vec)))
mean(mean(pi - pi_vec))

