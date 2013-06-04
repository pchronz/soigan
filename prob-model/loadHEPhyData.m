function [X, d] = loadHEPhyData()
  hephy = dlmread('data/hephy/HEPHY-UIBK_soigan.csv', ',', 1, 1);
  % remove all rows with NaNs throughout all data sets
  hephy_nan_lines = sum(isnan(hephy), 2) > 0;
  hephy(hephy_nan_lines, :) = [];
  d = hephy(:, end - 1)';
  hephy = hephy(:, 2:end-2);
  hephy = normalizeData(hephy);
  [N, D] = size(hephy);
  X = zeros(D, 1, N);
  X(:, 1, :) = hephy';
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

