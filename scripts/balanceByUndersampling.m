function [D_1, classlabel_1, D_2, classlabel_2] = balanceByUndersampling(D1,c1,D2,c2)
  D_1=D1;
  classlabel_1=c1;
  D_2=D2;
  classlabel_2=c2;

  N_1=length(D_1(:,1));
  N_2=length(D_2(:,1));

  D=[];
  classlabel=[];
  if(N_2>N_1)
    N=N_1;
    D=D_2;
    classlabel=classlabel_2;
  else
    N=N_2;
    D=D_1;
    classlabel=classlabel_1;
  end
  sub_data=zeros(N,length(D(1,:)));
  sub_labels=zeros(N,1);
  for n=1:N
    sub_data(n,:)=D(round(rand(1)*(N-1)+1),:);
    sub_labels(n)=classlabel(round(rand(1)*(N-1)+1));
  endfor
  if(N_2>N_1)
    D_2=sub_data;
    classlabel_2=sub_labels;
  else
    D_1=sub_data;
    classlabel_1=sub_labels;
  end
  assert(length(D_1)==length(D_2));
  assert(length(classlabel_1)==length(classlabel_2));
endfunction

