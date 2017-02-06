% Smoothing of Signal Using HP Filter and Classfying Signal based % on trend data.
clc;
clear all;
close all;

% Featching data
p=load('mydata1.mat');
a=p.feature;
f=input('Enter the signal data index');
e=a(f);
g=e{1};
data=g';
[f,c] = hpfilter(data,10000);
c=c';

% Differnce of data.
c1=c(1:end-1);
c2=c(2:end);
c3=abs(c1-c2);

% Statisitcs of data
mean=mean(c3);
std=std(c3);

%Finding Spike
i=2.5;
x=mean+i*std;
y=c3(abs(c3)>x);

while(length(y)>=3 && i<=4)
    i=i+0.1;
    x=mean+i*std;
    y=c3(abs(c3)>x);
end
i
%Finding Index of Spike

[tf,loc]=ismember(c3,y);
idx=[1:length(c3)];
idx=idx(tf);
idx=idx(loc(tf));


%Validate index vaules as the following value might be a of the same spike
if length(idx)==2
    if (idx(2)-idx(1)==1) ||(idx(2)-idx(1)==2)
        idx=idx(1);
    end
elseif length(idx)==3
    if (idx(2)-idx(1)==1) ||(idx(2)-idx(1)==2)
        idx = idx(idx~=idx(2));
    elseif (idx(3)-idx(2)==1) ||(idx(3)-idx(2)==2)
        idx = idx(idx~=idx(2));
    end
end

if (length(idx)>=3)
    display('Signal has no spike and hence signal can be smoothed by using the trend data')
    %Using trend component of the hpfilter
    [final_data]=hpfilter(data);
else
    %Index of spike data
    idx2=[];
    for i=1:length(idx)
        idx2=horzcat(idx2,[idx(i)-1:idx(i)+1]);
    end
    final_data=zeros(1,length(data));
    idx
    length(idx)
    if length(idx)==1
        A=data(1:idx-1);
        final_data(1:idx-1)=ones(1,length(A)).*(sum(A)/length(A));
        B=data(idx+1:end);
        final_data(idx+1:end)=ones(1,length(B)).*(sum(B)/length(B));
        final_data(idx-1:idx+1)=data(idx-1:idx+1);
    elseif length(idx)==2
        A=data(1:idx(1)-1);
        final_data(1:idx(1)-1)=ones(1,length(A)).*(sum(A)/length(A));
        B=data(idx(1)+1:idx(2)-1);
        final_data(idx(1)+1:idx(2)-1)=ones(1,length(B)).*(sum(B)/length(B));
        C=data(idx(2)+1:end);
        final_data(idx(2)+1:end)=ones(1,length(C)).*(sum(C)/length(C));
        final_data(idx(1)-1:idx(1)+1)=data(idx(1)-1:idx(1)+1);
        final_data(idx(2)-1:idx(2)+1)=data(idx(2)-1:idx(2)+1);
        
    end
end




% PLotting data
figure;
subplot(1,3,1);
plot(1:1:length(g'),g);
title('Original Data');
subplot(1,3,2);
plot(1:1:length(g')-1,abs(c3));
title('Cyclic Component');
subplot(1,3,3);
plot(1:1:length(g'),final_data);
title('Smoothed data');
% if length(idx)==1
%     final_data=mean(data(1:idx-2))*ones(idx-2,2);
% elseif length(idx)== 2
%     final_data=mean(data(1:idx(1)-2))*ones(idx(1)-2,2);