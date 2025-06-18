function [save_x,save_lam]=arc_length_cont(fun,x0,lam0,d_lam,lamfinal,maxiters,xtol,ftol,epFD,maxiterNew)
if nargin<5
    lamfinal=infty;
end
if nargin<6
    maxiters=100;
end
if nargin<7
    xtol=1e-14;
end
if nargin<8
    ftol=1e-12;
end
if nargin<9
    epFD=1e-8;
end
if nargin<10
    maxiterNew=100;
end
%%%%%%%%%%%%%%%%
lam1=lam0+d_lam;
np=length(x0);
x00=newton_method(@(x) fun(x,lam0),x0,xtol,ftol,epFD,maxiterNew);
save_x(:,1)=x00;
save_lam(1)=lam0;
x11=newton_method(@(x) fun(x,lam1),x00,xtol,ftol,epFD,maxiterNew);
save_x(:,2)=x11;
save_lam(2)=lam1;
%
dx=x11-x00;
lam=2*lam1-lam0;
ds=sqrt(sum((dx).^2)+d_lam^2);
k=2;
%additional condition
x=2*x11-x00;
while lam>lam0 && k<maxiters && lam<lamfinal && x(1)<0 %additional condition
    k=k+1;
    x=2*x11-x00;
    lam=2*lam1-lam0;
    alpha=(x11-x00)'/ds;
    bhta=(lam1-lam0)/ds;
    Nfun=@(x,lam) alpha*(x-x11)+bhta*(lam-lam1)-ds; %arc-length condition
    augfun=@(xlam) [fun(xlam(1:np,1),xlam(np+1,1));Nfun(xlam(1:np,1),xlam(np+1,1))];
    xlam0=[x;lam];
    [xlam,err,iterNew,J_x_lam]=newton_method(augfun,xlam0,xtol,ftol,epFD,maxiterNew);
    x=xlam(1:np,1);
    lam=xlam(np+1,1);
    save_x(:,k)=x;
    save_lam(k)=lam;
    lam0=lam1;
    lam1=lam;
    x00=x11;
    x11=x;
    fprintf('ARCLength ---- iter=%d, lambda=%2.4e \n', k,lam)
end
end