pkg load nan
more off

disp('reading in the data');
% load the csv: ',' as seperator, starting with the second line
data=dlmread('../data/creamce_with_sam_1_numerified_cropped.csv',',',1,0);

tic()
% seperate the inputs from the results
% TODO retaining only the numerical columns for now
D=data(:,1:13);
% using one class/type of test only for now
classlabel=data(:,17);
toc()

% remove all values that do not have a result associated with them
% TODO aggregation of values that lead up to a test result
idx=(classlabel>0);
D=D(idx,:);
classlabel=classlabel(idx);

disp('starting the classification');
CC=train_sc(D,classlabel,'svm');

%R=test_sc(CC,D);

