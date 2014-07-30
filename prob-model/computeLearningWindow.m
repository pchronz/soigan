function n = computeLearningWindow(F, tol, win_len)
  % DEBUG
  % F-Measure after learning
  %F = log([1:1000]);
  %F = F + 0.1*randn(1, length(F));
  % Smoothing does not seem to be necessary or do any good. 
  %% Smooth the series using a low pass filter.
  %F_ruff = F;
  %F = abs(ifft((fft(F)).*[ones(1, 200) zeros(1, length(F) - 230) ones(1, 30)])); 
  
  % Indices of the values on the outside of the tube.
  idx_out = find(double(abs(0.5*diff(F)) > tol));
  % Window lengths inside the tube; these are the distances between the indices of the values outside of the tube.
  win_lengths = diff(idx_out) .- 1;
  % Get the indices where the window length is sufficiently long. Get the first occurence.
  widx = find(double(win_lengths >= win_len));
  n = length(F);
  if(length(widx) > 0)
    n_idx = widx(1);
    n = idx_out(n_idx) + 1;
  endif

  % DEBUG Some plotting for verification.
  subplot(3, 1, 1)
  plot(F)
  subplot(3, 1, 2)
  plot(0.5*diff(F))
  subplot(3, 1, 3)
  plot(double(abs(0.5*diff(F)) > tol))
endfunction

