function [Z_n, z] = dec2oneofK(l, K, I)
    % transform the decimal number into the matrix Z_n with column vectors coded as 1-of-K
    % first create a matrix of column vectors representing all states
    k = (0:K-1)';
    k = k(:, ones(1, I));
    k = reshape(dec2base(k, K), K, I);
    % then create a matrix representing the state of each vector K times
    z = dec2base(l-1, K, I);
    z = z(ones(K, 1), :);
    % no compare both matrices to find out where the right states are selected i.e. which of all the states (k) is selected (z)
    Z_n = z == k;
endfunction

