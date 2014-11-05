more off
clear

pkg load statistics
pkg load nan
pkg load parallel

D = 10;
I = 6;
K = 3;
N = 20;

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
disp('Computing posterior...')
p_Z = computePosteriorApproximateVectorized(mus, Sigmas, pi, rho, X, d, K);
toc()

tic()
disp('Maximising Sigmas vectorized...')
[Sigmas_vec] = maxSigmasVectorized(X, mus, p_Z, D, K, I, N);
toc()

tic()
disp('Maximising Sigmas sequentially...')
[Sigmas] = maxSigmasSlow(X, mus, p_Z, D, K, I, N);
toc()

mean(mean(mean(mean(Sigmas - Sigmas_vec))))

