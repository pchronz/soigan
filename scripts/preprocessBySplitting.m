% aggregation of values that lead up to a test result
function [D, classlabel] = preprocessBySplitting(D_raw, classlabel_raw, w)
  % get all available targets
  idx = [1 : length(classlabel_raw)](classlabel_raw != 0);

  % compute the starting values for the respective windows
  idx_pre = @bsxfun(@minus, idx, w);
  % remove the values which do not have enough historic data
  idx(idx_pre<1) = [];
  idx_pre(idx_pre<1) = [];
  assert(length(idx) == length(idx_pre));

  % TODO XXX use reshape or something instead of a loop!
  D = zeros(length(D_raw(1, :)), w + 1, length(idx));
  for it = 1:length(idx)
    D(:, :, it) = D_raw(idx_pre:idx, :)';
  endfor
  [dim, N, M] = size(D);
  assert(dim == length(D_raw(1, :)));
  assert(N == w + 1);
  assert(M == length(idx));
endfunction

