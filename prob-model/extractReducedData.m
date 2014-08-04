function X_red = extractReducedData(X, services, dims)
  [D, I, N] = size(X);
  % Target array
  X_red = zeros(sum(dims)(1), length(services), N);
  for i = 1:length(services)
    X_red(:, i, :) = X(logical(dims(:, i)), services(i), :);
  endfor
endfunction

