function [p_0, p_1] = predictBaseline(X_next, Centers, rho)
  [I, D, K] = size(Centers);
  [D, I] = size(X_next);
  % identify the nearest cluster
  % XXX just testing...
  diffs = flipdim(rotdim(Centers, 1, [1, 2]), 1) - X_next(:, :, ones(1, K));
  % common cluster variant...
  % [D, K] = size(centers);
  % [D, I] = size(X_next);
  % % identify the nearest cluster
  % diffs = rotdim(centers(:, :, ones(1, I)), 1, [2, 3]) - X_next(:, :, ones(1, K));
  distances = zeros(I, K);
  clusters = zeros(1, I);
  for i = 1:I
    for k = 1:K
      distances(i, k) = diffs(:, i, k)' * diffs(:, i, k);
    endfor
    [d, idx] = min(distances(i, :));
    clusters(i) = idx;
  endfor

  % get the probability for the resulting cluster
  l = base2dec(num2str(clusters-1), K) + 1;
  p_1 = 0;
  if(sum(double(rho(:, 1) == l)) == 0)
    % this is needed since the relation of number of states observed
    % and the total possible number of states is out of proportion 
    % once we start to investigate more machines (greater I) and/or more
    % states per machine. In this case the best thing we can do is to pick
    % one of the observed patterns which have the greatest similiarity
    % i.e. the least number of different states per machine. It's a shot in
    % the blue, but it works.
    % TODO calculate the development of the relation observed states/total states
    % TODO reason why this distance metric does make sense; research related
    % distance metrics in information theory
    disp('The pattern to be predicted has not been observed in the past. Choosing the nearest neighbour instead...')
    p_1 = getNextObservedNeighbour(l, rho, K, I);
  else
    p_1 = rho(rho(:, 1) == l, 2);
  endif

  p_0 = 1-p_1;
endfunction

function p = getNextObservedNeighbour(l, rho, K, I)
  % (l_address, distance)
  nearest = zeros(1, 2);
  nearest(1, 1) = rho(1, 1);
  nearest(1, 2) = Inf;
  for rho_idx = rho(:, 1)'
    rho_base = dec2base(uint64(rho_idx - 1), K, I);
    l_base = dec2base(uint64(l - 1), K, I);
    dist = sum(double(rho_base == l_base));
    if(dist < nearest(1, 2))
      nearest(1, 1) = rho_idx;
      nearest(1, 2) = dist;
    endif
  endfor
  disp('Nearest neighbour has a distance of...')
  nearest(1, 2)
  p = rho(rho(:, 1) == nearest(1, 1), 2);
endfunction

