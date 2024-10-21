function main
clc;
close all;
clear all;

%%
% X in range [a,b] with probability distribution f1
a=-5;
b=5;
mux=0;
sigma_xsq=1;
f1=@(xv) ((1/sqrt(2*pi*sigma_xsq))*exp(-(xv-mux).^2/(2*sigma_xsq)));
f1_t=integral(f1,a,b);
f1=@(xv) f1(xv)/f1_t;
%%
% parameters

Mval=[2 4 8];

Mval=[16];

p_bval=[ 0.2 0.3 0.49];
% p_bval=[0.07];

nsam=[300 300 40 25 37];
nsam=[25];
%%
% initializing parameters to store values
numlocopt=zeros(length(Mval),1);
endist=zeros(length(Mval),length(p_bval)); % encoder distortion for all M and bit error rate
dedist=zeros(length(Mval),length(p_bval)); % decoder distortion for all M and bit error rate 
xmall=zeros(Mval(end)+1,length(Mval),length(p_bval)); % quantizers for all M and bit error rate 
exitm=zeros(length(Mval),length(p_bval)); % exitflag for all M and bit error rate
xinitm=zeros(Mval(end)+1,length(Mval),length(p_bval)); % initialization used for all M and bit error rate

%%
% computing constraint matrices for all M
A_all=zeros(Mval(end)-2,Mval(end)-1,length(Mval));
for M=Mval
A=[];
b1=[];
if M>2
A= zeros(M-1-1,M-1);
A(1,:)=[1 -1 zeros(1,M-1-2)];
for i=2:M-1-1
    A(i,:)=circshift(A(i-1,:),1);
end
b1=zeros(size(A,1),1);
end
A_all(1:(M-2),1:(M-1),find(M==Mval))=A;
end


[xmall,ymall,dmall,Mv]=max_quant();
%%
% main
for M=Mval % loop over M
    xsamp=[ linspace(-3+eps,3-eps,nsam(find(M==Mval))) ];

% deterministic initializations

xminit=nchoosek(xsamp,M-1);
xminit=[ones(size(xminit,1),1)*a xminit ones(size(xminit,1),1)*b];
rn=100;
xminit=xminit(randi(size(xminit,1),1,min(size(xminit,1),100)),:);
xminit(1,2:end-1)=linspace(-2,2,M-1);
xminit(2,2:end-1)=xmall(find(M==Mv),1:M-1);
xminit(3,:)=linspace(a,b,M+1);
xminit(4,:)=[linspace(a,b-(b-a)/2,M) b];
xminit(5,:)=[a linspace(a+(b-a)/2,b,M)];
xminit(6,:)=[a linspace(a+(b-a)/3,b,M)];
xminit(7,:)=[linspace(a,b-(b-a)/3,M) b];
for n1=min(size(xminit,1),400):rn
xminit(n1,:)=[a sort(a+(b-a)*rand(1,M-1)) b];
end
rn=size(xminit,1);
xrandinit=zeros(M+1,rn); % all initializations
xrm=zeros(M+1,rn); % final quantizer values for all initializations
erm=zeros(1,rn); % encoder distortions for all initializations
yrm=zeros(M,rn); % final quantizer representative values for all initializations
drm=zeros(1,rn); % decoder distortions for all initializations
exitflag=zeros(1,rn);
derend=zeros(M-1,rn);


for p_b=p_bval
    pbind=find(p_b==p_bval);

    p_err=1-(1-p_b)^(log(M)/log(2)); % symbol error
    c_1=p_err/(M-1);
    c_2=1-M*c_1;
    loopind=find(M==Mval); % loop index for M
    % constraints
    A=[];
    b1=[];
    if M>2
        A=A_all(1:(M-2),1:(M-1),(find(M==Mval))); % from the computation before
        b1=zeros(size(A,1),1); 
    end
    lb=[a*ones(M-1,1)];
    ub=[b*ones(M-1,1)];
    endist_rn=zeros(rn,1); % encoder distortions for each initialization
    dedist_rn=zeros(rn,1); % decoder distortions for each initialization
    xrn=zeros(M+1,rn); % quantizer for given M for each initialization
    xinitrn=zeros(M+1,rn); % quantizer initializations for given M for each initialization
    exitrn=zeros(rn,1); % exit flags for given M for each initialization
    tic
    for r=1:rn % loop over initializations
        disp(strcat('M=',num2str(M),', iteration=',num2str(r),', bit error rate=',num2str(p_b))); % display all parameters 
        xrandinit(:,r)=xminit(r,:)';
        x0=xminit(r,:);
        x1=x0;
        x1=sort(x1')';
        x0=sort(x0')'
        x0=x0(2:end-1); % optimizing only decision values that are not boundaries
        fun=@(x)f22fn(x,f1,a,b,p_err,c_1,c_2,mux); % objective function
        options = optimoptions('fmincon','MaxFunctionEvaluations',90000000,'MaxIterations',90000000,'SpecifyObjectiveGradient',true,'CheckGradients',true,'Display','none');
        [x,fval,exitflag,output,lambda,grad,hessian]=fmincon(fun,x0,A,b1,[],[],lb,ub,[],options); % gradient descent
        x=[a; x'; b]; % gradient descent output
        xrm(:,r)=x;
        erm(r)=fval;
        yrm(:,r)=reconstruction(x,f1,p_err,c_1,c_2,mux);
        ym=yrm(:,r);
        drm(r)=decoderdistortion(x,ym,f1,p_err,a,b);
        exitrn(r)=exitflag; % exit flag 
        xm=x ;% quantizer 
        ym=reconstruction(xm,f1,p_err,c_1,c_2,mux); % reconstruction levels
        [dist_enc]=encoderdistortion(xm,ym,f1,p_err,a,b); % encoder distortion
        [dist_dec]=decoderdistortion(xm,ym,f1,p_err,a,b); % decoder distortion
        endist_rn(r)=dist_enc; % encoder distortions for each initialization
        dedist_rn(r)=dist_dec; % decoder distortions for each initialization
        xrn(:,r)=xm; % quantizer for given M for each initialization
        % xrnall(:,1:M+1,loopind,r,pbind)=xm; % quantizers for all M and bit error rate for all iterations
        xinitrn(:,r)=x1; % quantizer initializations for given M for each initialization
        % xrninitall(:,1:M+1,loopind,r,pbind)=x1; % quantizer initializations for all M and bit error rate for all iterations
        % enall(loopind,r,pbind)=dist_enc; % encoder distortions for all M and bit error rate, all intializations
        [in1,in2]=min(endist_rn(1:r)); % finding minimum encoder distortion within the valid runs 
    xm=xrm(:,in2(1))
    ym=reconstruction(xm,f1,p_err,c_1,c_2,mux)
    dist_enc=encoderdistortion(xm,ym,f1,p_err,a,b)
    dist_dec=decoderdistortion(xm,ym,f1,p_err,a,b)
    ctr=1;
    cmpx=[xrm(:,in2(1))];
    cmp=in1;
    indx=zeros(1,length(erm));
    indx(in2(1))=1;
    for i=1:length(erm)
        if sum(sum(bsxfun(@power,bsxfun(@minus,cmpx,xrm(:,i)),2))./norm(xrm(:,i)) < 10^-5)==0
            cmpx=[cmpx xrm(:,i)];
            indx(i)=1;
            cmp=[cmp erm(i)];
            ctr=ctr+1;
        end
    end
    numlocopt(1)=ctr;
    exitcond=find(exitrn~=1)

    
    disp(strcat('Results for M=',num2str(M),', bit error rate=',num2str(p_b))); % display result for M
    disp('encoder and decoder distortions for all initializations:')
    endist_rn'
    dedist_rn'
    disp('quantizer:')
    x_opt=xrn(:,in2(1))' % optimum quantizer for current M
    y_opt=reconstruction(x_opt,f1,p_err,c_1,c_2,mux)
    disp('encoder distortion:')
    e_opt=endist_rn(in2(1)) % optimum encoder distortion for current M
    disp('decoder distortion:')
    d_opt=dedist_rn(in2(1)) % optimum decoder distortion for current M

    endist(loopind,pbind)=endist_rn(in2(1)); % encoder distortion for all M 
    dedist(loopind,pbind)=dedist_rn(in2(1)); % decoder distortion for all M 
    xmall(1:M+1,loopind,pbind)=xrn(:,in2(1)); % quantizers for all M 
    exitm(loopind,pbind)=exitrn(in2(1)); % exitflag for all M 
    % xinitm(1:M+1,loopind,pbind)=xrninitall(1:M+1,loopind,in2(1)); % initialization used for all M 
    % save(strcat('xcubed_noisy_fmincon_data','M',num2str(M),'pb',num2str(p_b),'.mat'),'x_opt','y_opt','e_opt','d_opt','p_b','M','xmall','endist','dedist','exitm','xinitm','xrnall','xrninitall','enall');
    save(strcat('gaussian_xycubed_fmincon_M',num2str(M),'pb',num2str(p_b),'a',num2str(a),'b',num2str(b),'.mat'),'xm','ym','dist_enc','dist_dec','erm','xrm','yrm','drm','xminit','exitrn','numlocopt','cmp','cmpx','indx','p_b','ctr');

        
        % [in1,in2]=min(endist_rn(1:r)); % finding minimum encoder distortion within the valid runs 
        % xm=xrm(:,in2(1))
        % ym=reconstruction(xm,f1,p_err,c_1,c_2,mux)
        % dist_enc=encoderdistortion(xm,ym,f1,p_err,a,b)
        % dist_dec=decoderdistortion(xm,ym,f1,p_err,a,b)
        % save(strcat('gaussian_xcubedy_fmincon_M',num2str(M),'pb',num2str(p_b),'a',num2str(a),'b',num2str(b),'.mat'),'xm','ym','dist_enc','dist_dec','erm','xrm','yrm','drm','xminit','exitrn','p_b');

    end
    toc
    % for r=1:rn
    % xrnall(1:M+1,loopind,r,pbind)=xrn(:,r);
    % xrninitall(1:M+1,loopind,r,pbind)=xinitrn(:,r);
    % end
    % indl=find(exitrn==1); % finding valid runs of gradient descent using exit flag values
    [in1,in2]=min(endist_rn); % finding minimum encoder distortion within the valid runs 
    xm=xrm(:,in2(1))
    ym=reconstruction(xm,f1,p_err,c_1,c_2,mux)
    dist_enc=encoderdistortion(xm,ym,f1,p_err,a,b)
    dist_dec=decoderdistortion(xm,ym,f1,p_err,a,b)
    ctr=1;
    cmpx=[xrm(:,in2(1))];
    cmp=in1;
    indx=zeros(1,length(erm));
    indx(in2(1))=1;
    for i=1:length(erm)
        if sum(sum(bsxfun(@power,bsxfun(@minus,cmpx,xrm(:,i)),2))./norm(xrm(:,i)) < 10^-5)==0
            cmpx=[cmpx xrm(:,i)];
            indx(i)=1;
            cmp=[cmp erm(i)];
            ctr=ctr+1;
        end
    end
    numlocopt(1)=ctr;
    exitcond=find(exitrn~=1)

    
    disp(strcat('Results for M=',num2str(M),', bit error rate=',num2str(p_b))); % display result for M
    disp('encoder and decoder distortions for all initializations:')
    endist_rn'
    dedist_rn'
    disp('quantizer:')
    x_opt=xrn(:,in2(1))' % optimum quantizer for current M
    y_opt=reconstruction(x_opt,f1,p_err,c_1,c_2,mux)
    disp('encoder distortion:')
    e_opt=endist_rn(in2(1)) % optimum encoder distortion for current M
    disp('decoder distortion:')
    d_opt=dedist_rn(in2(1)) % optimum decoder distortion for current M

    endist(loopind,pbind)=endist_rn(in2(1)); % encoder distortion for all M 
    dedist(loopind,pbind)=dedist_rn(in2(1)); % decoder distortion for all M 
    xmall(1:M+1,loopind,pbind)=xrn(:,in2(1)); % quantizers for all M 
    exitm(loopind,pbind)=exitrn(in2(1)); % exitflag for all M 
    % xinitm(1:M+1,loopind,pbind)=xrninitall(1:M+1,loopind,in2(1)); % initialization used for all M 
    % save(strcat('xcubed_noisy_fmincon_data','M',num2str(M),'pb',num2str(p_b),'.mat'),'x_opt','y_opt','e_opt','d_opt','p_b','M','xmall','endist','dedist','exitm','xinitm','xrnall','xrninitall','enall');
    save(strcat('gaussian_xycubed_fmincon_M',num2str(M),'pb',num2str(p_b),'a',num2str(a),'b',num2str(b),'.mat'),'xm','ym','dist_enc','dist_dec','erm','xrm','yrm','drm','xminit','exitrn','numlocopt','cmp','cmpx','indx','p_b','ctr');


end
end
% save(strcat('locoptpb',num2str(p_b),'.mat'),'numlocopt','p_b','Mval','rn','xrninitall','enall');
% save(strcat('xcubed_noisy_fmincon1_data.mat'),'p_bval','Mval','xmall','endist','dedist','exitm','xinitm','xrnall','xrninitall','enall');

function [ym]=reconstruction(xm,f1,p_err,c_1,c_2,mux)
M=length(xm)-1;
f2=@(xv) xv*f1(xv);
ym=zeros(1,M);
for i=1:M
    num=integral(f2,xm(i),xm(i+1),'ArrayValued',true);
    den=integral(f1,xm(i),xm(i+1),'ArrayValued',true);
    ym(i)=(c_1*mux+c_2*num)/(c_1+c_2*den);
end

function [dist_dec]=decoderdistortion(xm,ym,f1,p_err,a,b)
M=length(xm)-1;
c1=p_err/(M-1);
c2=1-M*c1;
dist_dec=0;
for i=1:M 
    f5=@(xv) (xv-ym(i))^2*f1(xv);
    dist_dec=dist_dec+c2*integral(f5,xm(i),xm(i+1),'ArrayValued',true);
    dist_dec=dist_dec+c1*integral(f5,a,b,'ArrayValued',true);
end

function [dist_enc]=encoderdistortion(xm,ym,f1,p_err,a,b)
M=length(xm)-1;
c1=p_err/(M-1);
c2=1-M*c1;
dist_enc=0;
for i=1:M 
    f5=@(xv) (xv-ym(i)^3)^2*f1(xv);
    dist_enc=dist_enc+c2*integral(f5,xm(i),xm(i+1),'ArrayValued',true);
    dist_enc=dist_enc+c1*integral(f5,a,b,'ArrayValued',true);
end

function [der]=derivative(xm,ym,f1,i,p_err)
M=length(xm)-1;
c1=p_err/(M-1);
c2=1-M*c1;
der=0;
    der=c2*(xm(i)-ym(i-1)^3)^2*f1(xm(i));
    der=der-c2*(xm(i)-ym(i)^3)^2*f1(xm(i));
    f3_1=@(xv) 3*ym(i-1)^2*(xv-ym(i-1)^3)*f1(xv);
    f3_2=@(xv) 3*ym(i)^2*(xv-ym(i)^3)*f1(xv);

    if xm(i-1)~=xm(i)
        dyixi=c2*f1(xm(i))*(xm(i)-ym(i-1))/(c1+c2*integral(f1,xm(i-1),xm(i)));
        der=der-2*c2*dyixi*integral(f3_1,xm(i-1),xm(i),'ArrayValued',true);
    end
    if xm(i)~=xm(i+1)
        dyi1xi=-c2*f1(xm(i))*(xm(i)-ym(i))/(c1+c2*integral(f1,xm(i),xm(i+1)));
        der=der-2*c2*dyi1xi*integral(f3_2,xm(i),xm(i+1),'ArrayValued',true);
    end
    
    der=der-2*c1*dyixi*integral(f3_1,xm(1),xm(end),'ArrayValued',true);
    der=der-2*c1*dyi1xi*integral(f3_2,xm(1),xm(end),'ArrayValued',true);

function [f22,der] = f22fn(x,f1,a,b,p_err,c_1,c_2,mux)

M=length(x)+1;
x=[a;x';b];
[ym]=reconstruction(x,f1,p_err,c_1,c_2,mux);
f22=encoderdistortion(x,ym,f1,p_err,a,b);
der=zeros(length(x),1);
for i=2:M
der(i)=derivative(x,ym,f1,i,p_err);
end
der=der(2:M);


function [xmall,ymall,dmall,Mval]=max_quant()
clc;
close all;
clear all;
Mval=[2 4 8 16 32];
xmall=zeros(length(Mval),max(Mval)-1);
ymall=zeros(length(Mval),max(Mval));
dmall=zeros(length(Mval),1);
M=2;
xmall(find(M==Mval),1:M-1)=0;
ymall(find(M==Mval),1:M)=[-0.7980 0.7980];
dmall(find(M==Mval))=0.3634;
M=4;
temp=[0.9816];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.4528 1.510];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
dmall(find(M==Mval))=0.1175;
M=8;
temp=[0.5006 1.050 1.748];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.2451 0.7560 1.344 2.152];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
dmall(find(M==Mval))=0.03454;
M=16;
temp=[0.2582 0.5224 0.7996 1.099 1.437 1.844 2.401];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.1284 0.3881 0.6568 0.9424 1.256 1.618 2.069 2.733];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
dmall(find(M==Mval))=0.009497;
M=32;
temp=[0.1320 0.2648 0.3991 0.5359 0.6761 0.8210 0.9718 1.130 1.299 1.482 1.682 1.908 2.174 2.505 2.977];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.06590 0.1981 0.3314 0.4668 0.6050 0.7473 0.8947 1.049 1.212 1.387 1.577 1.788 2.029 2.319 2.692 3.263];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
dmall(find(M==Mval))=0.002499;