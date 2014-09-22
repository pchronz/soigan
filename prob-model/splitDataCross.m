function [X_tr, d_tr, X_test, d_test] = splitDataCross(X, d, s, S)
  [D, I, N] = size(X);
  % split the data into training and validation sets
  % first split the data by target classes
  flags_0 = (d == 0);
  flags_1 = !flags_0;
  idx_0 = (1:N)(flags_0);
  idx_1 = (1:N)(flags_1);
  % 0s first
  assert(sum(flags_0) >= 2)
  parts = partitionSet(idx_0, S);
  idx_0_test = parts{s};
  % Just deal with it.
  idx_0_train_idx = [1:S];
  idx_0_train_idx(s) = [];
  idx_0_train = parts{idx_0_train_idx};
  % 1s second
  assert(sum(flags_1) >= 2)
  parts = partitionSet(idx_1, S);
  idx_1_test = parts{s};
  % Just deal with it.
  idx_1_train_idx = [1:S];
  idx_1_train_idx(s) = [];
  idx_1_train = parts{idx_1_train_idx};
  % now assemble both into common training and test sets
  X_test = X(:, :, [idx_0_test, idx_1_test]);
  d_test = d([idx_0_test, idx_1_test]);
  X_tr = X(:, :, [idx_0_train, idx_1_train]);
  d_tr = d([idx_0_train, idx_1_train]);
endfunction

function v_parts = partitionSet(v, S)
  N = length(v);
  min_els = floor(N/S);
  rest_els = mod(N, S);
  lens = min_els*ones(1, S);
  lens(1:rest_els) = lens(1:rest_els) + 1;
  lens = cumsum([0 lens]);
  v_parts = cell(S);
  for s = 1:S
    v_parts(s) = v(lens(s) + 1:lens(s + 1));
  endfor
  % XXX Test
  tot_len = 0;
  for s = 1:S
    tot_len = tot_len + length(v_parts{s});
  endfor
  assert(tot_len == length(v))
endfunction


