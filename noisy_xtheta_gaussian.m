function main
clc;
close all;
clear all;

%% parameters

%variance
sigma_thsq=1; % variance theta
sigma_xsq=1; % variance of source X
rhoval=[ -0.9 -0.5 0 0.5 0.9]; %correlation
 rhoval=[ 0 0.5]; %correlation
 rhoval=[ 0.9 0.5 0 -0.5 -0.9]; %correlation
p_bval=[0.05 0.07 0.2 0.3 0.49];
Mval=[8];


numsamp=[15 15 15 17 33];%number of samples on x for the grid in [-3,3]
numsamp=[30 30 30 30];
% numsamp=[17];%number of samples on x for the grid in [-3,3]
rn=10; % number of initializations: form a grid then pick rn number of initializations randomly 

%%
% discretizing theta, theta in [at,bt], mean mut, variance sigma_thsq
at=-5;
bt=5;
% thval=linspace(at+(bt-at)/(2*nt),bt-(bt-at)/(2*nt),nt); 
mut=0;

thval1=linspace(at,mut-2*sigma_thsq,1);
thval2=linspace(mut-2*sigma_thsq,mut-sigma_thsq,5);
thval3=linspace(mut-sigma_thsq,mut+sigma_thsq,11);
thval4=linspace(mut+sigma_thsq,mut+2*sigma_thsq,5);
thval5=linspace(mut+2*sigma_thsq,bt,1);
thval=[thval1(2:end) thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)];
thval=[thval1 thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)]';
% thval=[-2;-1;0;1;2];
nt=length(thval);
% pdf of theta
pth=zeros(1,length(thval));
f12=@(tv) ((1/sqrt(2*pi*sigma_thsq))*exp(-(tv-mut).^2/(2*sigma_thsq)));
sct=integral(f12,at,bt,'ArrayValued',true);
pth(1)=integral(f12,at,thval(1)+(thval(2)-thval(1))/2,'ArrayValued',true)/sct;
for i=2:length(thval)-1
    pth(i)=integral(f12,thval(i)-(thval(i)-thval(i-1))/2,thval(i)+(thval(i+1)-thval(i))/2,'ArrayValued',true)/sct;
end
pth(length(thval))=integral(f12,thval(end)-(thval(end)-thval(end-1))/2,bt,'ArrayValued',true)/sct;


encdistM=zeros(length(Mval),1);
decdistM=zeros(length(Mval),1);



a=-5;
b=5;

% 
mux=0; % mean of source X
fx=@(xv) ((1/sqrt(2*pi*sigma_xsq))*exp(-(xv-mux).^2/(2*sigma_xsq)));  
scalx=integral(@(xv) fx(xv),a,b);
fx=@(xv) fx(xv)./scalx;
[xmall,ymall]=max_quant();

for p_b=p_bval
    pbind=find(p_b==p_bval);


for M=Mval
        p_err=1-(1-p_b)^(log(M)/log(2)); % symbol error
    c1=p_err/(M-1);
    c2=1-M*c1;

m=find(M==Mval);
xnon=xmall(m,1:M-1);
xnon=[a xnon b];
% ym=ymall(m,1:M);
% [dist_dec_xnon]=decoderdistortionwotheta(xnon,ym,fx);
xsamp=[a linspace(-3,3,numsamp(find(M==Mval))) b];

A=[];
    b1=[];
    if M>2
        A=zeros((M-1-1)*length(thval),(M-1)*length(thval));
        A1=[1 -1 zeros(1,M-1-2+(M-1)*(length(thval)-1))];
        i=0;
        for j=1:length(thval)
        A(i+1,:)=A1;
        i1=i+1;
        for i=(j-1)*(M-1-1)+2:(j-1)*(M-1-1)+M-2
            A(i,:)=circshift(A(i-1,:),1);
        end
        if length(i)==0
            i=i1;
        end
        A1=circshift(A(i,:),2);
        end
        b1=zeros(size(A,1),1);
    end
    A;
    b1;


lb1=[a*ones(M-1,1)];
lb=repmat(lb1,length(thval),1);
ub1=[b*ones(M-1,1)];
ub=repmat(ub1,length(thval),1);

x0init=nchoosek(xsamp(2:end-1),M-1);
x0init=[a*ones(size(x0init,1),1) x0init b*ones(size(x0init,1),1)];
xrn1=randi(size(x0init,1),rn,length(thval));
xminit=zeros(rn,length(thval),M+1);
xminit(1,:,:)=repmat(xnon,length(thval),1);

for r=2:rn
xminit(r,:,:)=x0init(xrn1(r,:)',:);
end

% rn=size(xminit,1); % number of initializations 

xrandinit=zeros(length(thval),M+1,rn); % all initializations
xrm=zeros(length(thval),M+1,rn); % final quantizer values for all initializations
erm=zeros(1,rn); % encoder distortions for all initializations
yrm=zeros(M,rn); % final quantizer representative values for all initializations
drm=zeros(1,rn); % decoder distortions for all initializations
exitflagrn=zeros(1,rn);
derend=zeros(length(thval),M-1,rn);


for rho=rhoval
mux_corr=mux+rho*(sigma_xsq/sigma_thsq)^(1/2)*(thval(:)-mut); % mean of X conditional on theta 
sigma_xsq_corr=(1-rho^2)*sigma_xsq; % variance of X conditional on theta 
f1=@(xv,i) ((1/sqrt(2*pi*sigma_xsq_corr))*exp(-(xv-mux_corr(i)).^2/(2*sigma_xsq_corr)))*pth(i); % pdf of X conditional on theta

xm=[a*ones(length(thval),1) b*ones(length(thval),1)];
ym=reconstruction(xm,thval,f1,c1,c2,mux);
dist_enc_nr=encoderdistortion(xm,ym,f1,thval,c1,c2,a,b);
dist_dec_nr=decoderdistortion(xm,ym,f1,thval,c1,c2,a,b);

xm=repmat(xnon,length(thval),1);
ym=reconstruction(xm,thval,f1,c1,c2,mux);
dist_enc_fr=encoderdistortion(xm,ym,f1,thval,c1,c2,a,b);
dist_dec_fr=decoderdistortion(xm,ym,f1,thval,c1,c2,a,b);

% save(strcat('M',num2str(M),'dense',num2str(nt),'rho',num2str(rho),'fnr.mat'),'dist_dec_xnon','xm','ym',"dist_dec_fr","dist_enc_fr",'dist_enc_nr','dist_dec_nr','M','rho')
for r=1:rn
xrandinit(:,:,r)=xminit(r,:,:);
xm=reshape(xminit(r,:,:),length(thval),M+1);
xminitder=xm;

x0=xminitder;
x1=x0;
x1=sort(x1')';
x0=sort(x0')';
x0=x0(:,2:end-1); % optimizing only decision values that are not boundaries
x0=x0';
x0=x0(:);
fun=@(x)f22fn(x,thval,f1,a,b,c1,c2,mux,r); % objective function
% options = optimoptions('fmincon','MaxFunctionEvaluations',90000000,'MaxIterations',90000000,'Display','iter','PlotFcn',{@optimplotx,@optimplotfval,@optimplotfirstorderopt});
options = optimoptions('fmincon','MaxFunctionEvaluations',90000000,'MaxIterations',90000000,'SpecifyObjectiveGradient',true,'CheckGradients',true,'Display','off');
tic
[x,fval,exitflag,output,lambda,grad,hessian]=fmincon(fun,x0,A,b1,[],[],lb,ub,[],options); % gradient descent
toc
disp(strcat('Results for M=',num2str(M),', bit error rate=',num2str(p_b),', rho=',num2str(rho),'r=',num2str(r))); % display result for M
   
x=[a*ones(length(thval),1) reshape(x,M-1,length(thval))' b*ones(length(thval),1)]; % gradient descent output
% disp("fmincon result");
% x;
y=reconstruction(x,thval,f1,c1,c2,mux);
exitflagrn(r)=exitflag;
dist_enc_fmin=encoderdistortion(x,y,f1,thval,c1,c2,a,b);
dist_dec_fmin=decoderdistortion(x,y,f1,thval,c1,c2,a,b);

xm=x;
xrm(:,:,r)=xm;
yrm(:,r)=y;
erm(r)=dist_enc_fmin;
drm(r)=dist_dec_fmin;

[in1 in2]=min(erm(1:r));
xm=xrm(:,:,in2);
ym=reconstruction(xm,thval,f1,c1,c2,mux);
dist_enc=encoderdistortion(xm,ym,f1,thval,c1,c2,a,b);
dist_dec=decoderdistortion(xm,ym,f1,thval,c1,c2,a,b);

save(strcat('noisy_xthetaM',num2str(M),'rho',num2str(rho),'varth',num2str(sigma_thsq),'varx',num2str(sigma_xsq),'thval',num2str(nt),'pb',num2str(p_b),'noiseless_xtheta_gaussian.mat'),'thval','xm','ym','dist_enc','dist_dec','erm','xrm','yrm','drm','xrandinit','exitflagrn','sigma_thsq','sigma_xsq','dist_enc_nr','dist_dec_nr','dist_enc_fr','dist_dec_fr')

end

[in1 in2]=min(erm);
xm=xrm(:,:,in2);
ym=reconstruction(xm,thval,f1,c1,c2,mux);
dist_enc=encoderdistortion(xm,ym,f1,thval,c1,c2,a,b);
dist_dec=decoderdistortion(xm,ym,f1,thval,c1,c2,a,b);

save(strcat('noisy_xthetaM',num2str(M),'rho',num2str(rho),'varth',num2str(sigma_thsq),'varx',num2str(sigma_xsq),'thval',num2str(nt),'pb',num2str(p_b),'noiseless_xtheta_gaussian.mat'),'thval','xm','ym','dist_enc','dist_dec','erm','xrm','yrm','drm','xrandinit','exitflagrn','sigma_thsq','sigma_xsq','dist_enc_nr','dist_dec_nr','dist_enc_fr','dist_dec_fr')

end
end
end

% function d=dist_decquant(xm,ym,fx)
% M=length(ym);
% d=0;
% for i=1:M
%     d=d+integral(@(xv) (xv-ym(i)).^2.*fx(xv),xm(i),xm(i+1));
% end

% function xm=enc(ym,a,b)
% M=length(ym);
% T1=[a; ym(1:M-1); b];
% T2=[a; ym(2:M); b];
% xm=(T1+T2)/2;
% xm=xm';

% function ym=dec(xm,fx)
% M=length(xm)-1;
% ym=zeros(M,1);
% for i=1:M
%     fx1= @(xv) xv.*fx(xv);
%         num=integral(fx1,xm(i),xm(i+1));
%         den=integral(@(xv) fx(xv),xm(i),xm(i+1));
%     ym(i)=num/den;
% end


function [dist_dec]=decoderdistortion(xthetam,ym,f1,thval,c1,c2,a,b)
M=size(xthetam,2)-1;
dist_dec=0;
for i=1:M
    for k=1:length(thval)
        f1temp= @(xv) f1(xv,k);
        f5=@(xv) (xv-ym(i))^2*f1temp(xv);
        dist_dec=dist_dec+c2*integral(f5,xthetam(k,i),xthetam(k,i+1),'ArrayValued',true);
        dist_dec=dist_dec+c1*integral(f5,a,b,'ArrayValued',true);
    end
end

% function [dist_dec]=decoderdistortionwotheta(xm,ym,fx)
% M=size(xm,2)-1;
% dist_dec=0;
% for i=1:M
%         f5=@(xv) (xv-ym(i))^2*fx(xv);
%         dist_dec=dist_dec+integral(f5,xm(i),xm(i+1),'ArrayValued',true);
% end

function [dist_enc]=encoderdistortion(xthetam,ym,f1,thval,c1,c2,a,b)
M=size(xthetam,2)-1;
dist_enc=0;
for i=1:M
    for k=1:length(thval)
        f1temp= @(xv) f1(xv,k);
        f5=@(xv) (xv+thval(k)-ym(i))^2*f1temp(xv);
        dist_enc=dist_enc+c2*integral(f5,xthetam(k,i),xthetam(k,i+1),'ArrayValued',true);
        dist_enc=dist_enc+c1*integral(f5,a,b,'ArrayValued',true);
    end
end


function [ym]=reconstruction(xthetam,thval,f1,c1,c2,mux)
M=size(xthetam,2)-1;

ym=zeros(1,M);
for i=1:M
    num=0;
    den=0;
    for j=1:length(thval)
        f1temp= @(xv) f1(xv,j);
        f2=@(xv) xv*f1temp(xv);
        num=num+integral(f2,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
        den=den+integral(f1temp,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
    end
    % if den~=0
        ym(i)=(c1*mux+c2*num)/(c1+c2*den);
    % else
    %     ym(i)=(1/size(xthetam,1))*sum(xthetam(:,i));
    % end
end

function [der]=derivative(xm,ym,f1,i,t,thval,c1,c2,a,b)
M=size(xm,2)-1;

der=0;
den1=0;
den2=0;
num=f1(xm(t,i),t);
for th=1:size(thval)
    den1=den1+integral(@(xv) f1(xv,th),xm(th,i-1),xm(th,i));
    den2=den2+integral(@(xv) f1(xv,th),xm(th,i),xm(th,i+1));
end
% den1=den1;
% den2=den2;



    der=c2*(xm(t,i)+thval(t)-ym(i-1))^2*f1(xm(t,i),t);
    der=der-c2*(xm(t,i)+thval(t)-ym(i))^2*f1(xm(t,i),t);

dyixi=c2*(xm(t,i)-ym(i-1))*num/(c1+c2*den1);
dyi1xi=-c2*(xm(t,i)-ym(i))*num/(c1+c2*den2);
for th=1:length(thval)
    f3_1=@(xv) (xv+thval(th)-ym(i-1))*f1(xv,th);
    f3_2=@(xv) (xv+thval(th)-ym(i))*f1(xv,th);
     der=der-2*c1*dyixi*integral(f3_1,a,b,'ArrayValued',true);
     der=der-2*c1*dyi1xi*integral(f3_2,a,b,'ArrayValued',true);
    if xm(th,i-1)~=xm(th,i)        
        der=der-2*c2*dyixi*integral(f3_1,xm(th,i-1),xm(th,i),'ArrayValued',true);
    end
    if xm(th,i)~=xm(th,i+1)        
        der=der-2*c2*dyi1xi*integral(f3_2,xm(th,i),xm(th,i+1),'ArrayValued',true);
    end
end   
   
    
function [f22,der] = f22fn(x,thval,f1,a,b,c1,c2,mux,r)

M=length(x)/length(thval)+1;
x=[a*ones(length(thval),1) reshape(x,M-1,length(thval))' b*ones(length(thval),1)];
[ym]=reconstruction(x,thval,f1,c1,c2,mux);
der=zeros(length(thval),M-1);
for i=2:M
    for t=1:length(thval)
[der(t,i-1)]=derivative(x,ym,f1,i,t,thval,c1,c2,a,b);
    end
end

der=der';
der=der(:);


x=x';
x=x(:);

f22=0;
for i=1:M
    for t=1:length(thval)
            f22=f22+c2*integral(@(xv)(xv+thval(t)-ym(i))^2*f1(xv,t),x((t-1)*(M+1)+i),x((t-1)*(M+1)+i+1),'ArrayValued',true);
  f22=f22+c1*integral(@(xv)(xv+thval(t)-ym(i))^2*f1(xv,t),a,b,'ArrayValued',true);

    end
end



function [xmall,ymall]=max_quant() 
%max quantization table from...
%zero mean, variance 1 gaussian
Mval=[2 4 8 16 32];
xmall=zeros(length(Mval),max(Mval)-1);
ymall=zeros(length(Mval),max(Mval));
M=2;
xmall(find(M==Mval),1:M-1)=0;
ymall(find(M==Mval),1:M)=[-0.7980 0.7980];
M=4;
temp=[0.9816];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.4528 1.510];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
M=8;
temp=[0.5006 1.050 1.748];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.2451 0.7560 1.344 2.152];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
M=16;
temp=[0.2582 0.5224 0.7996 1.099 1.437 1.844 2.401];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.1284 0.3881 0.6568 0.9424 1.256 1.618 2.069 2.733];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
M=32;
temp=[0.1320 0.2648 0.3991 0.5359 0.6761 0.8210 0.9718 1.130 1.299 1.482 1.682 1.908 2.174 2.505 2.977];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.06590 0.1981 0.3314 0.4668 0.6050 0.7473 0.8947 1.049 1.212 1.387 1.577 1.788 2.029 2.319 2.692 3.263];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
