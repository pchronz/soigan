% aggregation of values that lead up to a test result
function [D, classlabel] = preprocessByAggregation(D_raw, classlabel_raw)
  idx=(classlabel_raw>0);
  lin=[1:length(idx)];
  idx=lin(idx!=0);
  % TODO XXX do without a loop
  D_agg=zeros(length(idx), length(D_raw(1,:)));
  classlabel_agg=zeros(length(idx),1);
  i=1;
  l_m1=1;
  for l=idx,
    max_window=l-l_m1;
    w=min(20, max_window);
    D_agg(i,:)=mean(D_raw(l-w+1:l,:));
    classlabel_agg(i)=classlabel_raw(l);
    l_m1=l;
    i=i+1;
  endfor
  D=D_agg;
  classlabel=classlabel_agg;
endfunction

