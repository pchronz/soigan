function [Centers, rho] = learnBaselineModel(K, X, d)
  [D, I, N] = size(X);
  Centers = zeros(I, D, K);
  states = zeros(I, N);
  disp('Clustering... ')
  for i = 1:I
    i
    empty = true;
    idx = 0;
    centers = 0;
    while(empty)
      try 
        [idx, centers] = kmeans(reshape(X(:, i, :), D, N)', K);
        empty = false;
      catch
        i
        disp('Got an empty cluster, trying again...');
      end_try_catch
    endwhile
    states(i, :) = idx';
    Centers(i, :, :) = centers';
  endfor

  % TODO optimize for performance: if N < K^I then iterate over all K^I
  % else get the observed states and only iterate over them
  % convert the states to representative numbers
  L = zeros(1, N);
  for n = 1:N
    state = states(:, n);
    L(1, n) = base2dec(strcat(numbase2str(state - 1)), K) + 1;
    if(K < 10)
      foo = base2dec(strcat(num2str(state - 1)'), K) + 1;
      assert(foo == L(1, n))
    endif
  endfor
  % remove duplicates
  L = unique(L);

  disp('Rhos')
  % allocating most of those values (especially for large K, I does not
  % make any sense, since most values will be unobserved. You will hit memory
  % constraints very quickly. Instead only allocate memory for those rhos which
  % actually have been observed: those in L
  % (rho index/address, probability)
  rho = zeros(length(L), 2);
  rho(:, 1) = L';
  for l = L
    l_state = strbase2double(dec2base(uint64(l-1), K, I)) + 1;
    if(K < 10)
      foo = str2double(dec2base(uint64(l-1), K, I))' + 1;
      assert(l_state == foo)
    endif
    l_observed = prod(states == l_state(:, ones(1, N)), 1);
    if(sum(l_observed) > 0)
      assert(sum(double(rho(:, 1) == l)) == 1)
      rho(rho(:, 1) == l, 2) = sum(d(logical(l_observed))) / sum(l_observed);
    endif
  endfor
  % XXX common cluster variant...
  % [D, I, N] = size(X);
  % % first do a clustering on the whole monitoring data
  % % run k-means to obtain an initial estimate for the mixture components (mean and covariance)
  % Xk = reshape(X, D, I*N, []);
  % [idx, centers] = kmeans(Xk', K);
  % centers = centers';
  % % reshape the idx vector to conform with X's format (1xIxN)
  % states = reshape(idx', I, N);
  % rho = 0.5 * ones(K^I, 1);
  % for l = 1:K^I
  %   % XXX this will only work for 1 < K < 10
  %   l_state = str2num(dec2base(l-1, K, I)') + 1;
  %   l_observed = prod(states == l_state(:, ones(1, N)));
  %   if(sum(l_observed) > 0)
  %     rho(l) = sum(d(logical(l_observed))) / sum(l_observed);
  %   endif
  % endfor
endfunction

function str = numbase2str(num)
  str = blanks(length(num));
  syms = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];
  for n = 1:length(num)
    str(n) = syms(num(n) + 1);
  endfor
endfunction

function dou = strbase2double(str)
  dou = zeros(length(str), 1);
  syms = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];
  for s = 1:length(str)
    dou(s, 1) = [1:36](syms == str(s)) - 1;
  endfor
endfunction


