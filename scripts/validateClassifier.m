function [e_total, e_1, e_2] = validateClassifier(CC, D_test, classlabel_test)
  % use a separate out-of-sample data set for final validation
  disp('starting the validation run');
  R=test_sc(CC, D_test);

  % output the results
  error_v=classlabel_test'!=R.classlabel;
  idx1=(classlabel_test==1);
  errors_1=sum(error_v(idx1));
  idx2=(classlabel_test==2);
  errors_2=sum(error_v(idx2));
  errors_total=sum(error_v)
  N_1_test=length(error_v(idx1));
  N_2_test=length(error_v(idx2));
  e_total = errors_total/(N_1_test + N_2_test);
  e_1 = errors_1/N_1_test;
  e_2 = errors_2/N_2_test;
  disp('total error rate'), disp(e_total)
  disp('error rate for class 1 (false failure)'), disp(e_1)
  disp('error rate for class 2 (false success)'), disp(e_2)
endfunction

