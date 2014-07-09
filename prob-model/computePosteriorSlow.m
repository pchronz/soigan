function p_Z = computePosteriorSlow(mus, Sigmas, pi, rho, X, d, K)
    [D, D, K, I] = size(Sigmas);
    [D, I, N] = size(X);
    p_Z = zeros(K^I, N);
    % DEBUG
    mvn = zeros(N, K^I*I);
    for n = 1:N
      n
      for l = 1:K^I
        [Z_n, z] = dec2oneOfK(l, K, I);
        % select the right mus
        mus_l = zeros(D, I);
        % z_idx = (base2dec(z(1, :)', K)) + 1;
        z_idx = zeros(I, 1);
        for i = 1:I
          z_idx(i) = base2decfast(z(1, i), K) + 1;
        endfor
        % assert(z_idx == z_idx_fast)
        for i = 1:I
          mus_l(:, i) = mus(:, z_idx(i), i);
        endfor
        % select the right Sigmas
        Sigmas_l = zeros(D, D, I);
        for i = 1:I
          Sigmas_l(:, :, i) = Sigmas(:, :, z_idx(i), i);
        endfor
        % select the right pis
        pi_l = pi(logical(Z_n));
        % compute the posterior for the current state and observation
        p_Z(l, n) = rho(l)^d(n) * (1 - rho(l))^(1 - d(n));
        for i = 1:I
          p_x_n_i = mvnpdf(X(:, i, n)', mus_l(:, i)', Sigmas_l(:, :, i));
          p_Z(l, n) = p_Z(l, n) * pi_l(i) * p_x_n_i;
        endfor
      endfor
    endfor
    % normalize
    %p_Z = p_Z ./ sum(p_Z);
    %p_Z = e.^(log(p_Z) .- log(sum(p_Z)));
    %for n = 1:N
    %  if(sum(p_Z(:, 1)) >= 1.00001 || sum(p_Z(:, 1)) <= 0.99999)
    %    p_Z(:, 1)
    %    n
    %    1 - sum(p_Z(:, 1))
    %    error('p_Z does not sum to one')
    %  endif
    %endfor
endfunction

