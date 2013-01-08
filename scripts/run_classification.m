pkg load nan
more off

disp('reading in the data');
% load the csv: ',' as seperator, starting with the second line
data=dlmread('../data/creamce_with_sam_1.csv',',',1,0);

tic()
% seperate the inputs from the results
% TODO retaining only the numerical columns for now
D=data(:,1:13);
% using one class/type of test only for now
classlabel=data(:,17);
toc()

% XXX column 6 has all values 0, so let's kick it out, since it makes the normalization impossible
D=D(:,[1:5,7:end]);

% remove all values that do not have a result associated with them
% TODO aggregation of values that lead up to a test result
% UNKNOWN
idx=(classlabel>0);
D=D(idx,:);
classlabel=classlabel(idx);
% MISSING
idx=(classlabel!=3);
D=D(idx,:);
classlabel=classlabel(idx);

% XXX merge OK and WARNING
idx=(classlabel==4);
classlabel(idx)=1;

% TODO normalize the data
m=mean(D);
sigma=var(D);
D=bsxfun(@minus,D,m);
D=bsxfun(@rdivide,D,sigma);

disp('starting the classification');
tic();
% TODO try out different classifiers
CC=train_sc(D,classlabel);
toc();

R=test_sc(CC,D);

