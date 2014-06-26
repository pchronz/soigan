function expanded = expandDataSet(X, D_X, N, D)
    assert(min(eig(cov(X))) > 0)
    expanded = rand(N, D) - 1/2;
    expanded(:, 1:D_X) = X;
    assert(min(eig(cov(expanded))) > 0)
endfunction

