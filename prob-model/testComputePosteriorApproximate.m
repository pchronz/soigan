more off
clear

pkg load statistics
pkg load nan
pkg load parallel

global para = false;

D = 10;
I = 7;
K = 3;
N = 5;

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
disp('Approximate parallel')
p_Z_appr_vec = computePosteriorApproximateVectorized(mus, Sigmas, pi, mat2cell(rho, ones(K^I, 1)), X, d, K);
toc()

tic()
disp('Approximate')
p_Z_appr = computePosteriorApproximate(mus, Sigmas, pi, rho, X, d, K);
toc()

tic()
disp('Exact sequential')
p_Z_slo = computePosteriorSlow(mus, Sigmas, pi, rho, X, d, K);
toc()

tic()
disp('Exact parallel')
p_Z_vec = computePosteriorVectorized(mus, Sigmas, pi, rho, X, d, K);
toc()

diffs = zeros(N, K^I);
for n = 1:N
  for l = 1:K^I
    idx = p_Z_appr_vec{n}(1, :) == l;
    if(sum(idx) == 0)
      diffs(n, l) = abs(p_Z_appr(l, n) - 0);
    else
      diffs(n, l) = abs(p_Z_appr(l, n) - p_Z_appr_vec{n}(2, idx));
    endif
  endfor
endfor
mean(mean(diffs))
mean(mean(abs(p_Z_appr - p_Z_slo)))

