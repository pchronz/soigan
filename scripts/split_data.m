% split the data into a test and a training set according to ratio_test
% the resulting data sets elicit approximately the same ratio between classlabels as the original set
function [D_train, classlabel_train, D_test, classlabel_test] = split_data(D, classlabel, ratio_test, pos, balance)
  % split the data by class labels
  % by this we ensure that both the test set and the training set will contain both classes
  % TODO make this a loop over all available classes
  idx=(classlabel==1);
  D_1=D(idx,:);
  classlabel_1=classlabel(idx);
  idx=(classlabel==2);
  D_2=D(idx,:);
  classlabel_2=classlabel(idx);

  if(balance)
    % TODO experiment with both under- and oversampling
    [D_1, classlabel_1, D_2, classlabel_2] = balanceByOversampling(D_1, classlabel_1, D_2, classlabel_2);
  end

  % split the data into a training and a test set
  N_1=length(D_1(:,1));
  N_1_test=round(N_1*ratio_test);
  N_1_end=min(pos*N_1_test, N_1);
  D_1_test=D_1((pos-1)*N_1_test+1:N_1_end,:);
  D_1((pos-1)*N_1_test+1:N_1_end,:)=[];
  classlabel_1_test=classlabel_1((pos-1)*N_1_test+1:N_1_end);
  classlabel_1((pos-1)*N_1_test+1:N_1_end)=[];
  D_1_train=D_1;
  classlabel_1_train=classlabel_1;

  N_2=length(D_2(:,1));
  N_2_test=round(N_2*ratio_test);
  N_2_end=min(pos*N_2_test, N_2);
  D_2_test=D_2((pos-1)*N_2_test+1:N_2_end,:);
  D_2((pos-1)*N_2_test+1:N_2_end,:)=[];
  classlabel_2_test=classlabel_2((pos-1)*N_2_test+1:N_2_end);
  classlabel_2((pos-1)*N_2_test+1:N_2_end)=[];
  D_2_train=D_2;
  classlabel_2_train=classlabel_2;

  % now merge both classes again into the test and training sets
  D_test=[D_1_test;D_2_test];
  classlabel_test=[classlabel_1_test;classlabel_2_test];
  D_train=[D_1_train;D_2_train];
  classlabel_train=[classlabel_1_train;classlabel_2_train];
endfunction

