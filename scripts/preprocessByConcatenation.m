% aggregation of values that lead up to a test result
function [D, classlabel] = preprocessByConcatenation(D_raw, classlabel_raw, w)
  % get all available targets
  idx = [1 : length(classlabel_raw)](classlabel_raw != 0);

  % compute the starting values for the respective windows
  idx_pre = @bsxfun(@minus, idx, w);
  % remove the values which do not have enough historic data
  idx(idx_pre<1) = [];
  idx_pre(idx_pre<1) = [];
  assert(length(idx) == length(idx_pre));

  dimension = length(D_raw(1, :)) + 1;
  D = zeros(length(idx), (w + 1) * dimension - 1);
  classlabel = zeros(length(idx), 1);
  it = 1;
  for i = idx_pre
    d = zeros(1, length(D(1, :)));
    it_2 = 1;
    for j = [i : i + w - 1]
      d((it_2 - 1) * dimension + 1 : it_2 * dimension - 1) = D_raw(j, :);
      d(it_2 * dimension) = classlabel_raw(j);
      it_2 = it_2 + 1;
    endfor
    j = i + w;
    d((it_2 - 1) * dimension + 1 : it_2 * dimension - 1) = D_raw(j, :);
    D(it, :) = d;
    classlabel(it) = classlabel_raw(j);
    it = it + 1;
  endfor
endfunction

