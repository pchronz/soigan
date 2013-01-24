pkg load nan
more off

disp('reading in the data');
% only read in the file if the data variable does not already exist
if(exist('data') == 0)
  % load the csv: ',' as seperator, starting with the second line
  data=dlmread('../data/creamce_with_sam_1.csv',',',1,0);
end

tic()
% seperate the inputs from the results
% and kick out the timestamps
% TODO retaining only the numerical columns for now
D=data(:,2:13);
% using one class/type of test only for now
classlabel=data(:,17);
toc()

% XXX column 5 has all values 0, so let's kick it out, since it makes the normalization impossible
D=D(:,[1:4,6:end]);

% XXX merge OK and WARNING
idx=(classlabel==4);
classlabel(idx)=1;

% TODO aggregation of values that lead up to a test result
idx=(classlabel>0);
lin=[1:length(idx)];
idx=lin(idx!=0);
% TODO XXX do without a loop
D_agg=zeros(length(idx), length(D(1,:)));
classlabel_agg=zeros(length(idx),1);
i=1;
l_m1=1;
for l=idx,
  max_window=l-l_m1;
  w=2;
  if(w > max_window),
    w=max_window;
  endif
  % for some reason it does not work if I replace 2 with w. WTF?
  D_agg(i,:)=mean(D(l-3+1:l,:));
  classlabel_agg(i)=classlabel(l);
  l_m1=l;
  i=i+1;
endfor
D=D_agg;
classlabel=classlabel_agg;

% remove all values that do not have a result associated with them
% UNKNOWN
idx=(classlabel>0);
D=D(idx,:);
classlabel=classlabel(idx);
% MISSING
idx=(classlabel!=3);
D=D(idx,:);
classlabel=classlabel(idx);

% normalize the data
m=mean(D);
sigma=var(D);
D=bsxfun(@minus,D,m);
D=bsxfun(@rdivide,D,sigma);

% split the data by class labels
% by this we ensure that both the test set and the training set will contain both classes
% TODO make this a loop over all available classes
idx=(classlabel==1);
D_1=D(idx,:);
classlabel_1=classlabel(idx);
idx=(classlabel==2);
D_2=D(idx,:);
classlabel_2=classlabel(idx);

% split the data in a training and a test set
% tr=^"test ratio"
% 20<->80 split (test<->train) is a widely used convention [GMNS2009, CB2008]
tr=0.2;
N_1=length(D_1(:,1));
N_1_test=round(N_1*tr);
D_1_test=D_1(1:N_1_test,:);
classlabel_1_test=classlabel_1(1:N_1_test,:);
D_1_train=D_1(N_1_test+1:end,:);
classlabel_1_train=classlabel_1(N_1_test+1:end,:);

N_2=length(D_2(:,1));
N_2_test=round(N_2*tr);
D_2_test=D_2(1:N_2_test,:);
classlabel_2_test=classlabel_2(1:N_2_test,:);
D_2_train=D_2(N_2_test+1:end,:);
classlabel_2_train=classlabel_2(N_2_test+1:end,:);

% now merge both classes again into the test and training sets
D_test=[D_1_test;D_2_test];
classlabel_test=[classlabel_1_test;classlabel_2_test];
D_train=[D_1_train;D_2_train];
classlabel_train=[classlabel_1_train;classlabel_2_train];

% TODO try the following scheme:
% 1 undersampling of the majority class to obtain a balanced set
% 2 train multiple (how many?) SVMs to obtain various experts
% 3 create a mixture of experts using the SVMs
% TODO try using the RVM
% TODO do cross validation
% TODO split the training set into a training and evaluation set
% TODO optimize the parameters using simulated annealing or another nonlinear optimization technique
% this should incorporate a higher penalty to represent the gravity of false successes

disp('starting the training');
tic();
% XXX thought about trying PCA, but this really does not help all that much here.
% PCA is not really helpful for classification because it does not take take the 
% labels into account and thus may be a hindrance to the classification
% Fisher's linear discriminant is the way to go since it provides a projection which
% does take the labels into account
% TODO try out different classifiers
num_runs=1000;
errors=zeros(num_runs,1);
models={};
MODE.TYPE='rbf';
for it=1:num_runs
  % TODO implement cross-validation as another inner loop
  % train
  MODE.hyperparameter.c_value=rand(1)*250;
  MODE.hyperparameter.gamma=rand(1)/10000;
  CC=train_sc(D_train,classlabel_train, MODE);
  models(it)=CC;

  % test
  % TODO test with test set within cross-validation
  R=test_sc(CC,D_train);
  errors(it)=sum(classlabel_train'!=R.classlabel);
endfor
toc();

% choose the best model found through cross validation
[e,i]=sort(errors);
CC=models{i(1)};


% TODO use a separate out-of-sample data set for final validation
disp('starting the validation run');
R=test_sc(CC, D_test);

% output the results
error_v=classlabel_test'!=R.classlabel;
idx1=(classlabel_test==1);
errors_1=sum(error_v(idx1));
idx2=(classlabel_test==2);
errors_2=sum(error_v(idx2));
errors_total=sum(error_v)
disp('total error rate'), disp(errors_total/(N_1_test + N_2_test))
disp('error rate for class 1'), disp(errors_1/N_1_test)
disp('error rate for class 2'), disp(errors_2/N_2_test)

