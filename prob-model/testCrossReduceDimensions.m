more off
clear
pkg load statistics
pkg load nan

[X, d] = loadGoeGridFullData(0, 500);
[services, dims] = crossReduceDimensions(X, d);

X_red = extractReducedData(X, services, dims);

