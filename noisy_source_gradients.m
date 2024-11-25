function main
clc;
close all;
clear all;
%% parameters
Mval=[2];
rho=0;

% discretizing theta, theta in [at,bt], mean mut, variance sigma_thsq
at=-5;
bt=5;
mut=0;
sigma_thsq=1;
thval1=linspace(at,mut-2*sigma_thsq,1);
thval2=linspace(mut-2*sigma_thsq,mut-sigma_thsq,2);
thval3=linspace(mut-sigma_thsq,mut+sigma_thsq,3);
thval4=linspace(mut+sigma_thsq,mut+2*sigma_thsq,2);
thval5=linspace(mut+2*sigma_thsq,bt,1);
thval=[thval1(2:end) thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)];
thval=[thval1 thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)]';
thval=[-1;1];
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

au=-5;
bu=5;
muu=0; % mean of noiseless source U
sigma_usq=1;
fu=@(uv) ((1/sqrt(2*pi*sigma_usq))*exp(-(uv-muu).^2/(2*sigma_usq))); % pdf of U 
scalu=integral(@(uv) fu(uv),au,bu);
fu=@(uv) fu(uv)./scalu;

aw=-5;
bw=5;
sigma_wsq=0.1;
muw=0; % mean of independent additive noise
fw=@(wv) ((1/sqrt(2*pi*sigma_wsq))*exp(-(wv-muw).^2/(2*sigma_wsq))); % pdf of W 
scalw=integral(@(wv) fw(wv),aw,bw);
fw=@(wv) fw(wv)/scalw;

fux=@(uv,xv) fu(uv).*fw(xv-uv);
f1=@(uv,xv,t) fux(uv,xv).*pth(t);

a=au+at;
b=bu+bt;

xsamp=linspace(a,b,12);

for M=Mval
    % initializations 
    x0init=nchoosek(xsamp(2:end-1),M-1);
    x0init=[a*ones(size(x0init,1),1) x0init b*ones(size(x0init,1),1)];
    rn=20;
    xrn1=randi(size(x0init,1),rn,length(thval));
    xminit=zeros(rn,length(thval),M+1);
    for r=1:rn
    xminit(r,:,:)=x0init(xrn1(r,:)',:);
    end
    xrandinit=zeros(length(thval),M+1,rn); % all initializations
    xrm=zeros(length(thval),M+1,rn); % final quantizer values for all initializations
    erm=zeros(1,rn); % encoder distortions for all initializations
    yrm=zeros(M,rn); % final quantizer representative values for all initializations
    drm=zeros(1,rn); % decoder distortions for all initializations
    exitflag=zeros(1,rn);
    derend=zeros(length(thval),M-1,rn);
    for r=1:rn
    flag=1;
    xmiter=zeros(length(thval),M+1,100); % quantizer values for each iteration given an initial point
    endist=zeros(1,100); % encoder distortions for each iteration given an initial point
    frendist=zeros(1,100); % fractional difference in encoder distortions for each iteration given an initial point
    dedist=zeros(1,100); % decoder distortions for each iteration given an initial point
    derv=zeros(length(thval),M-1,100);
    iter=1;
    xrandinit(:,:,r)=xminit(r,:,:);
    xmiter(:,:,1)=reshape(xminit(r,:,:),length(thval),M+1);
    xm=xmiter(:,:,1);
    ym=reconstruction(xm,thval,f1,au,bu);
    dist_enc=encoderdistortion(xm,ym,f1,thval,au,bu);
    dist_dec=decoderdistortion(xm,ym,f1,thval,au,bu);
    endist(1)=dist_enc;
    dedist(1)=dist_dec;
    delta=1;
    while flag
    for t=1:length(thval)
    for i=2:M
        der=derivative(xm,ym,f1,i,t,thval,au,bu);
        derv(t,i-1,iter)=der;
        temp=xm(t,i)-delta*der;
        xm1=xm;
        xm1(t,i)=temp;
        ym=reconstruction(xm1,thval,f1,au,bu);
        d1=encoderdistortion(xm1,ym,f1,thval,au,bu);

        if (temp>xm(t,i-1) && temp<xm(t,i+1)) && d1<dist_enc
            xm(t,i)=temp;
            
        else
            [xm]=check(xm,f1,delta,der,dist_enc,i,t,thval,au,bu);
        end
        ym=reconstruction(xm,thval,f1,au,bu);
        dist_enc=encoderdistortion(xm,ym,f1,thval,au,bu);
    end
    end
    xmtemp=xm;
    ymtemp=reconstruction(xmtemp,thval,f1,au,bu);
    dist_enctemp=encoderdistortion(xmtemp,ymtemp,f1,thval,au,bu);
    if iter>1
        if (endist(iter) == endist(iter-1))
            flag=0;
            exitflag(r)=2;
        end
    end
    if all(abs(derv(:,:,iter)) <10^-7 ) 
        flag=0;
        exitflag(r)=1;
    else
        iter=iter+1;
        xm=xmtemp;
        ym=ymtemp;
        xmiter(:,:,iter)=xm;
        dist_enc=dist_enctemp;
        endist(iter)=dist_enc;
        dedist(iter)=decoderdistortion(xm,ym,f1,thval,au,bu);
    end
    end
    derend(:,:,r)=derv(:,:,iter);
    xrm(:,:,r)=xm;
    erm(r)=dist_enc;
    yrm(:,r)=reconstruction(xm,thval,f1,au,bu);
    drm(r)=decoderdistortion(xm,yrm(:,r),f1,thval,au,bu);
    disp(strcat('M = ',num2str(M),', r = ',num2str(r),', rho = ',num2str(rho)))
    exitf=exitflag(r);
    xm
    ym
    dist_enc
end
[in1 in2]=min(erm);
xm=xrm(:,:,in2)
ym=reconstruction(xm,thval,f1,au,bu)
dist_enc=encoderdistortion(xm,ym,f1,thval,au,bu)
dist_dec=decoderdistortion(xm,ym,f1,thval,au,bu)

save(strcat('grad_xthetaM',num2str(M),'rho',num2str(rho),'noisysource_gaussian.mat'),'xm','ym','dist_enc','dist_dec','erm','xrm','yrm','drm','derend','xrandinit')
end



function [xm]=check(xm,f1,delta,der,dist_enc,i,t,thval,au,bu)
while delta>10^-7
    delta=delta/10;
    temp=xm(t,i)-delta*der;
    xm1=xm;
    xm1(t,i)=temp;
    ym=reconstruction(xm1,thval,f1,au,bu);
    d1=encoderdistortion(xm1,ym,f1,thval,au,bu);
    if (temp>xm1(t,i-1) && temp<xm1(t,i+1) ) && d1<dist_enc
        xm(t,i)=temp;
        break;
    end
end


function [dist_dec]=decoderdistortion(xthetam,ym,f1,thval,au,bu)
M=size(xthetam,2)-1;
dist_dec=0;
for i=1:M
    for k=1:length(thval)
        fux2=@(uv,xv) (uv-ym(i)).^2.*f1(uv,xv,k);
        dist_dec=dist_dec+integral2(fux2,au,bu,xthetam(k,i),xthetam(k,i+1));
    end
end



function [f22]=encoderdistortion(x,ym,f1,thval,au,bu)
M=size(x,2)-1;
x11=x;
[ym]=reconstruction(x,thval,f1,au,bu);


f22=0;
for i=1:M
    for t=1:length(thval)
            f22=f22+integral2(@(uv,xv)(uv+thval(t)-ym(i)).^2.*f1(uv,xv,t),au,bu,x(t,i),x(t,i+1));
    end
end



function [ym]=reconstruction(xthetam,thval,f1,au,bu)
M=size(xthetam,2)-1;
ym=zeros(1,M);
for i=1:M
    num=0;
    den=0;
    for j=1:length(thval)
        fux1= @(uv,xv) uv.*f1(uv,xv,j);
        num=num+integral2(fux1,au,bu,xthetam(j,i),xthetam(j,i+1),'AbsTol',0,'RelTol',1e-10);
        den=den+integral2(@(uv,xv) f1(uv,xv,j),au,bu,xthetam(j,i),xthetam(j,i+1),'AbsTol',0,'RelTol',1e-10);
    end
    if den~=0
    ym(i)=num/den;
    end
end



function [der]=derivative(x11,ym,fux,i,t1,thval,au,bu)
M=size(x11,2)-1;
der=0;

T3=0;
T23=0;
        T1=integral(@(uv) uv.*fux(uv,x11(t1,i-1),t1),au,bu);
        T2=ym(i-1)*integral(@(uv) fux(uv,x11(t1,i-1),t1),au,bu);
        for t=1:length(thval)
        T3=T3+integral2(@(uv,xv) fux(uv,xv,t),au,bu,x11(t,i-1),x11(t,i),'AbsTol',0,'RelTol',1e-10);
        T23=T23+integral2(@(uv,xv) fux(uv,xv,t),au,bu,x11(t,i),x11(t,i+1),'AbsTol',0,'RelTol',1e-10);
        end
        T21=integral(@(uv) uv.*fux(uv,x11(t1,i),t1),au,bu);
        T22=ym(i)*integral(@(uv) fux(uv,x11(t1,i),t1),au,bu);
        
    deryi=(T1-T2)/(T3);
    deryi1=-(T21-T22)/(T23);
    der=der+integral(@(uv)(uv+thval(t1)-ym(i-1)).^2.*fux(uv,x11(t1,i-1),t1),au,bu);
    der=der-integral(@(uv)(uv+thval(t1)-ym(i)).^2.*fux(uv,x11(t1,i),t1),au,bu);
        
    for t=1:length(thval)
        der=der-2*deryi*integral2(@(uv,xv)(uv+thval(t)-ym(i-1)).*fux(uv,xv,t),au,bu,x11(t,i-1),x11(t,i),'AbsTol',0,'RelTol',1e-10);
        der=der-2*deryi1*integral2(@(uv,xv)(uv+thval(t)-ym(i)).*fux(uv,xv,t),au,bu,x11(t,i),x11(t,i+1),'AbsTol',0,'RelTol',1e-10);        
    end

   