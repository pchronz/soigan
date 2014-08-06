more off
clear
pkg load statistics
pkg load nan
pkg load parallel

%[X, d] = loadGoeGridFullData(0, 500);
[X, d] = loadEwsData(1000);
tic()
[services, dims] = crossReduceDimensions(X, d, 10);
toc()

X_red = extractReducedData(X, services, dims);

