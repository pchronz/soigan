function [X, d] = createSyntheticData(N, I, R, D, K)
  % Ratio Ok/(Ok + NOk).
  ratio = 1.0
  % Redundancy groups.
  red_groups = zeros(1, I);
  while (ratio < 0.75 || ratio > 0.95)
    % Put the services into redundancy groups. 
    for i = 1:I
      red_groups(1, i) = randi(R);
    end
    % Generate the discrete data.
    [X_disc, d] = generateDiscreteData(N, I, R, D, K, red_groups);
    % Verify that we are satisfied with the ratio Ok/NOk.
    ratio = 1 - sum(d)/length(d)
  end

  % Get equally spaced means for the states.
  % TODO Add some noise to the means.
  % TODO Choose the means randomly.
  % TODO Get the variances.
  means = zeros(R, K, D);
  for r = 1:R
    for k = 1:K
      means(r, k, :) = k/(K + 1);
    end
  end

  % Generate the observations.
  X = zeros(D, I, N);
  mu = zeros(D, 1);
  for n = 1:N
    for i = 1:I
      % Get the redundancy group.
      r = red_groups(1, i);
      % Get the mean.
      mu(:, 1) = means(r, X_disc(i, n), :);
      % Sample the observation.
      % TODO Use a different distribution. Rather something like a multidimensional beta-distribution (not a Dirichlet though).
      X(:, i, n) = mvnrnd(mu', 0.01*eye(D));
    end
  end
endfunction

function [X_disc, d] = generateDiscreteData(N, I, R, D, K, red_groups)
  % Discrete data.
  X_disc = zeros(I, N);
  % Interpretation of states (Ok/NOk).
  states = zeros(R, K);
  % Generate states for each redundancy group.
  for r = 1:R
    % Decide which states are Ok and which are NOk.
    states = zeros(1, K);
    for k = 1:K
      states(r, k) = double(rand() > 0.7);
    end
    % Sample the states.
    idxs = 1:I;
    idxs = idxs(red_groups == r);
    for i = idxs
      for n = 1:N
        X_disc(i, n) = randi(K);
      end
    end
  end

  % Generate the global values.
  d = ones(1, N);
  for n = 1:N
    % If all of the services in one redundancy group are in any of the failed
    % states, than the global state is failed as well.
    % Get the states for the current time.
    x = X_disc(:, n);
    % Transform the states to Ok/NOk.
    x_state = zeros(1, I);
    for i = 1:I
      % Get the redundancy group.
      r = red_groups(1, i);
      % Check whether the states is Ok/NOk in that redundancy group.
      x_state(1, i) = states(r, x(i));
    end
    % Check whether any redundancy group is completely NOk.
    for r = 1:R
      % Get the services of that redundancy group.
      services = 1:I;
      services = services(red_groups(1, :) == r);
      % Get the states of the redundancy group.
      if (and(x_state(1, services)))
        d(1, n) = 0;
      end
    end
  end
endfunction


