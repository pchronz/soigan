function [D_1, classlabel_1, D_2, classlabel_2] = balanceByOversampling(D1,c1,D2,c2)
  D_1=D1;
  classlabel_1=c1;
  D_2=D2;
  classlabel_2=c2;

  N_1=length(D_1(:,1));
  N_2=length(D_2(:,1));

  D=[];
  classlabel=[];
  if(N_2<N_1)
    N=N_2;
    D=D_2;
    classlabel=classlabel_2;
  else
    N=N_1;
    D=D_1;
    classlabel=classlabel_1;
  end
  difference=abs(N_2-N_1);
  extra_data=zeros(difference,length(D(1,:)));
  extra_labels=zeros(difference,1);
  for n=1:difference
    extra_data(n,:)=D(round(rand(1)*(N-1)+1),:);
    extra_labels(n)=classlabel(round(rand(1)*(N-1)+1));
  endfor
  if(N_2<N_1)
    D_2=[D_2;extra_data];
    classlabel_2=[classlabel_2;extra_labels];
  else
    D_1=[D_1;extra_data];
    classlabel_1=[classlabel_1;extra_labels];
  end
  assert(length(D_1)==length(D_2));
endfunction

