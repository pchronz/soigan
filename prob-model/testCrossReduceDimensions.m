more off
clear
pkg load statistics
pkg load nan
pkg load parallel

[X, d] = loadGoeGridFullData(0, 500);
tic()
[services, dims] = crossReduceDimensions(X, d, max(8, nproc()));
toc()

X_red = extractReducedData(X, services, dims);

