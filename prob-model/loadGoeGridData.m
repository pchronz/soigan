function [X, d] = loadGoeGridData()
  goe = dlmread('data/goegrid/GoeGrid_soigan.csv', ',', 1, 1);
  % remove all rows with NaNs throughout all data sets
  goe_nan_lines = sum(isnan(goe), 2) > 0;
  goe(goe_nan_lines, :) = [];
  d = goe(:, end - 1)';
  goe = goe(:, 2:end-2);
  goe = normalizeData(goe);
  [N, D] = size(goe);
  X = zeros(D, 1, N);
  X(:, 1, :) = goe';
endfunction

function normalized = normalizeData(X)
  [N, D] = size(X);
  s = var(X);
  normalized = X(:, s > 0);
  normalized = normalized - mean(normalized)(ones(1, N), :);
  normalized = normalized * diag(1 ./ sqrt(var(normalized)));
  assert(sum(abs(mean(normalized))) < 0.0001);
  s2 = sum(var(normalized)) / size(normalized)(2);
  assert(s2 < 1.0001 && s2 > 0.9999);
endfunction

