function [X, d] = loadEwsData()
  apache = dlmread('data/ews/apache.csv', ';', 1, 1);
  % remove all rows with NaNs throughout all data sets
  apache_nan_lines = sum(isnan(apache), 2) > 0;
  jboss = dlmread('data/ews/jboss.csv', ';', 1, 1);
  jboss_nan_lines = sum(isnan(jboss), 2) > 0;
  localhost = dlmread('data/ews/localhost.csv', ';', 1, 1);
  localhost = localhost(:, [1, 2, 3, 4]);
  localhost_nan_lines = sum(isnan(localhost), 2) > 0;
  mysql = dlmread('data/ews/mysql.csv', ';', 1, 1);
  mysql_nan_lines = sum(isnan(mysql), 2) > 0;
  all_nan_lines = or(apache_nan_lines, jboss_nan_lines, localhost_nan_lines, mysql_nan_lines);
  apache(all_nan_lines, :) = [];
  jboss(all_nan_lines, :) = [];
  localhost(all_nan_lines, :) = [];
  mysql(all_nan_lines, :) = [];
  apache = normalizeData(apache);
  jboss = normalizeData(jboss);
  localhost = normalizeData(localhost);
  mysql = normalizeData(mysql);
  [N1, D1] = size(apache);
  [N2, D2] = size(jboss);
  [N3, D3] = size(localhost);
  [N4, D4] = size(mysql);
  max_N = max([N1, N2, N3, N4]);
  assert(N1 == max_N)
  assert(N2 == max_N)
  assert(N3 == max_N)
  assert(N4 == max_N)
  max_D = max([D1, D2, D3, D4]);
  X = zeros(max_D, 4, max_N);
  if(D1 < max_N)
    apache = expandDataSet(apache, D1, max_N, max_D);
  endif
  if(D2 < max_N)
    jboss = expandDataSet(jboss, D2, max_N, max_D);
  endif
  if(D3 < max_N)
    localhost = expandDataSet(localhost, D3, max_N, max_D);
  endif
  if(D4 < max_N)
    mysql = expandDataSet(mysql, D4, max_N, max_D);
  endif
  X(:, 1, :) = apache';
  X(:, 2, :) = jboss';
  X(:, 3, :) = localhost';
  X(:, 4, :) = mysql';
  workloadsla = dlmread('data/ews/workloadsla.csv', ';', 1, 1);
  d = workloadsla(:, 1)';
  d(all_nan_lines) = [];
  Nd = size(d)(2);
  assert(Nd == max_N)
endfunction

function expanded = expandDataSet(X, D_X, N, D)
    expanded = rand(N, D);
    expanded(:, 1:D_X) = X;
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

