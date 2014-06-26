function [X, d] = loadGoeGridFullData(delay)
  goe = dlmread('data/goegrid/goegrid.csv', ',', 1, 1);
  [N, foo] = size(goe);
  % re-align the monitoring values with the targets
  Idx = 1:N;
  d_start = Idx(logical(goe(:, end)))(1);
  d_end = Idx(logical(goe(:, end)))(end);
  X_start = d_start - delay;
  X_end = d_end - delay;
  goe = [goe(X_start:X_end, 1:end - 3), goe(d_start:d_end, end - 2 : end)];
  % remove all rows with NaNs throughout all data sets
  goe_nan_lines = sum(isnan(goe), 2) > 0;
  goe(goe_nan_lines, :) = [];
  % split the whole data set into the subsets for all machines
  goe1 = goe(:, 3:8);
  goe2 = goe(:, 10:15);
  goe3 = goe(:, 17:22);
  goe4 = goe(:, 24:29);
  goe5 = goe(:, 31:36);
  goe6 = goe(:, 38:43);
  goe7 = goe(:, 45:50);
  goe8 = goe(:, 52:57);
  goe9 = goe(:, 59:64);
  goe10 = goe(:, 66:71);
  goe11 = goe(:, 73:78);
  goe12 = goe(:, 80:85);
  goe13 = goe(:, 87:92);
  goe14 = goe(:, 94:99);
  goe15 = goe(:, 101:106);
  goe16 = goe(:, 108:113);
  goe17 = goe(:, 115:120);
  goe18 = goe(:, 122:127);
  goe19 = goe(:, 129:134);
  % Clean all the metrics
  goe1 = cleanMetrics(goe1);
  goe2 = cleanMetrics(goe2);
  goe3 = cleanMetrics(goe3);
  goe4 = cleanMetrics(goe4);
  goe5 = cleanMetrics(goe5);
  goe6 = cleanMetrics(goe6);
  goe7 = cleanMetrics(goe7);
  goe8 = cleanMetrics(goe8);
  goe9 = cleanMetrics(goe9);
  goe10 = cleanMetrics(goe10);
  goe11 = cleanMetrics(goe11);
  goe12 = cleanMetrics(goe12);
  goe13 = cleanMetrics(goe13);
  goe14 = cleanMetrics(goe14);
  goe15 = cleanMetrics(goe15);
  goe16 = cleanMetrics(goe16);
  goe17 = cleanMetrics(goe17);
  goe18 = cleanMetrics(goe18);
  goe19 = cleanMetrics(goe19);
  % normalize the data sets
  goe1 = normalizeData(goe1);
  goe2 = normalizeData(goe2);
  goe3 = normalizeData(goe3);
  goe4 = normalizeData(goe4);
  goe5 = normalizeData(goe5);
  goe6 = normalizeData(goe6);
  goe7 = normalizeData(goe7);
  goe8 = normalizeData(goe8);
  goe9 = normalizeData(goe9);
  goe10 = normalizeData(goe10);
  goe11 = normalizeData(goe11);
  goe12 = normalizeData(goe12);
  goe13 = normalizeData(goe13);
  goe14 = normalizeData(goe14);
  goe15 = normalizeData(goe15);
  goe16 = normalizeData(goe16);
  goe17 = normalizeData(goe17);
  goe18 = normalizeData(goe18);
  goe19 = normalizeData(goe19);
  [N, D] = size(goe1);
  X = zeros(D, 19, N);
  X(:, 1, :) = goe1';
  X(:, 2, :) = goe2';
  X(:, 3, :) = goe3';
  X(:, 4, :) = goe4';
  X(:, 5, :) = goe5';
  X(:, 6, :) = goe6';
  X(:, 7, :) = goe7';
  X(:, 8, :) = goe8';
  X(:, 9, :) = goe9';
  X(:, 10, :) = goe10';
  X(:, 11, :) = goe11';
  X(:, 12, :) = goe12';
  X(:, 13, :) = goe13';
  X(:, 14, :) = goe14';
  X(:, 15, :) = goe15';
  X(:, 16, :) = goe16';
  X(:, 17, :) = goe17';
  X(:, 18, :) = goe18';
  X(:, 19, :) = goe19';
  d = goe(:, end)';
  % Check for singular covariance matrices.
  for i = 1:19
    assert(isdefinite(cov(reshape(X(:, i, :), D, N)')) == 1)
  end
endfunction

