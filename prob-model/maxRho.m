function rho = maxRho(rho, K, I, p_Z, d, N)
  % rho
  rho = zeros(1, K^I);
  sum_p_Z = zeros(1, K^I);
  %for n = [1:N](logical(d))
  for n = 1:N
    p_Z_n = p_Z{n};
    idx_l = p_Z_n(1, :);
    for l = idx_l
      rho(l) += p_Z_n(2, p_Z_n(1, :) == l) * d(n);
      sum_p_Z(l) += p_Z_n(2, p_Z_n(1, :) == l);
    endfor
  endfor
  % In case we encounter a very improbable global state, assign 0.5 to the rho, since we just don't know any better. 
  rho = rho ./ sum_p_Z;
  % TODO Remove
  % Test whether all rho values are ok.
  if(sum(isnan(rho)) != 0)
    sum_p_Z
    min(sum_p_Z)
    p_Z
    pi
    d
    rho
  endif
  rho(isnan(rho)) = 0.5;
  % Are there any vaules outstide [0, 1]?
  idx = rho < 0;
  if(any(idx))
    rho(idx) = 0;
    warning('There are sub-zero rhos: capped them.')
  endif
  idx = rho > 1;
  if(any(idx))
    rho(rho > 1)
    rho(idx) = 1;
    warning('There are rhos greater than 1.0: capped them.')
  endif
end

