more off

% create the data first
% TODO keeping the monitoring data 1-dim for now
% D x I x N
N = 1000;
% mixture components
K = 2;
X_1 = normrnd(0.7, 0.05, 1, N/2);
X_2 = normrnd(0.3, 0.1, 1, N/2);
X = zeros(1, 1, N);
X(1, 1, :) = [X_1, X_2];
% the global system state observations for now: 1 if mixture k=1, 0 if mixture k=2
% TODO train specific combinations of mixtures to be 1 or 0
d = [zeros(1, N/2), ones(1, N/2)];

D = size(X, 1);
K = 2;
I = 1;

% starting the learning phase
% initialize the model parameters
mus = [0.1, 0.5];
Sigmas = [0.5, 0.3];
pi = [0.5; 0.5];
rho = 0.5 * ones(K^I, 1);

for it = 1:10
  it
  % E-step
  % compute the posterior for all possible states for Z for all observations N
  % this results in a matrix with (K^I)*N entries, you might want to optimize this at some point
  % nevertheless we will need the posterior many times, so it should help us quite a lot
  % let's get it over with...
  p_Z = zeros(K^I, N);
  for n = 1:N
    % iteration of the states in Z_n
    for l = 1:K^I
      % transform the decimal number into the matrix Z_n with column vectors coded as 1-of-K
      [Z_n, z] = dec2oneofK(l, K, I);
      p_d_n = rho(l)^d(n) * (1-rho(l))^(1-d(n));
      p_X_n = 1;
      for i = 1:I
        % select the right mixture component for this state vector
        mu_k = mus(str2num(z(1, i)) + 1);
        Sigma_k = Sigmas(str2num(z(1, i)) + 1);
        % compute the likelihood for the system observations and multiply by the total likelihood for all systems
        % XXX fix to multivariate normal
        % XXX what to do if this value becomes small beyond machine precision? This might happen when I becomes large.
        % ==> probably then this value does not matter anyway and we keep it at zero.
        p_X_n = p_X_n * normpdf(X(:, i, n), mu_k, Sigma_k);
      endfor
      % p_Z_n now
      % select the right components
      Pi = pi(:, ones(1, I));
      pi_l = Pi(Z_n);
      p_Z_n = prod(pi_l);
      p_Z(l, n) = p_d_n * p_X_n * p_Z_n;
    endfor
  endfor
  % normalize the values
  p_Z = p_Z ./ sum(p_Z)(ones(K^I, 1), :);

  % M-step
  % rho
  for l = 1:K^I
    rho(l) = sum(p_Z(l, :) * d') / sum(p_Z(l, :));
  endfor
  % pi
  for k = 1:K
    pi(k) = 0;
    for n = 1:N
      for l = 1:K^I
        [Z_n, z] = dec2oneofK(l, K, I);
        pi(k) = pi(k) + p_Z(l, n) * sum(Z_n(k, :), 2);
      endfor
    endfor
    pi(k) = 1/(I*N) * pi(k);
  endfor
  % mu
  for k = 1:K
    mus(k) = 0;
    for n = 1:N
      for l = 1:K^I
        [Z_n, z] = dec2oneofK(l, K, I);
        mus(k) = mus(k) + p_Z(l, n) * X(:, :, n) * diag(Z_n(k, :));
      endfor
    endfor
    % normalize
    C = 0;
    for n = 1:N
      for l = 1:K^I
        [Z_n, z] = dec2oneofK(l, K, I);
        C = C + p_Z(l, n) * sum(Z_n(k, :))
      endfor
    endfor
    C
    mus(k) = 1/C * mus(k);
  endfor
  % Sigma
  for k = 1:K
    Sigmas(k) = zeros(D);
    for n = 1:N
      for l = 1:K^I
        [Z_n, z] = dec2oneofK(l, K, I);
        X_n = X(:, :, n);
        % broadcast the mu_k vector 
        mu_ks = mus(k)(:, ones(1, I));
        dev = X_n - mu_ks;
        % the columns of the Kronecker product are the matrices of the outer products of the deviations
        E = reshape(eye(I), 1, I^2);
        Dev = kron(dev, dev);
        % only select the matrices where the inidices of the multiplied vectors are the same (i x i); quasi the diagonal matrices of the Kronecker product
        Dev = Dev(:, E);
        assert(size(Dev, 2) == I);
        % filter the relevant matrices the weights of the latent variable states for component k
        Dev = Dev(:, Z_n(k, :));
        % sum it up and reshape to obtain the unweighted (via posterior) result
        Sigma_data = reshape(sum(Dev, 2), D, D);

        Sigmas(k) = Sigmas(k) + p_Z(l, n) .* Sigma_data;
      endfor
    endfor
    % normalize
    C = 0;
    for n = 1:N
      for l = 1:K^I
        [Z_n, z] = dec2oneofK(l, K, I);
        C = C + p_Z(l, n) * sum(Z_n(k, :));
      endfor
    endfor
    Sigmas(k) = 1/C * Sigmas(k);
  endfor

  % TODO compute and monitor the likelihood

  mus
  Sigmas
  pi
endfor







