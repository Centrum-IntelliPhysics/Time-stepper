function [save_x,save_lam,save_eigs]=arc_length_cont_JFNKgmres(fun,x0,lam0,d_lam,lamfinal,maxiters,xtol,ftol,epFD,maxiterNew)
if nargin<5
    lamfinal=infty;
end
if nargin<6
    maxiters=100;
end
if nargin<7
    xtolJFNKgmres=1e-14;
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
eqfun=@(x,ep) x-fun(x,ep); %used for steady-states
stabfun=@(u0,ep,v,delta) (fun(u0+delta*v,ep)-fun(u0,ep))/delta; %used for eigenvalues
delta=1e-6;
lam1=lam0+d_lam;
np=length(x0);
xtolJFNKgmres=[xtol,xtol*100];
parms = [maxiters, maxiterNew, 0.9, 2, 20];
x00 = nsoli(x0,@(x) eqfun(x,lam0),xtolJFNKgmres,parms);
%x00=newton_method(@(x) fun(x,lam0),x0,xtol,ftol,epFD,maxiterNew);
save_x(:,1)=x00;
save_lam(1)=lam0;
save_eigs(:,1)=eigs(@(v) stabfun(x00,lam0,v,delta),np,[],3,'largestabs');
%x11=newton_method(@(x) fun(x,lam1),x00,xtol,ftol,epFD,maxiterNew);
x11 = nsoli(x00,@(x) eqfun(x,lam1),xtolJFNKgmres,parms);
save_x(:,2)=x11;
save_lam(2)=lam1;
save_eigs(:,2)=eigs(@(v) stabfun(x11,lam1,v,delta),np,[],3,'largestabs');
%
dx=x11-x00;
lam=2*lam1-lam0;
ds=sqrt(sum((dx).^2)+d_lam^2);
k=2;
%
flag_eigs=(nargout==3);
%additional condition
x=2*x11-x00;
every_k=100000; %every k NKGMRES, do a full newton (not needed but to be sure)
while k<maxiters 
    k=k+1;
    x=2*x11-x00;
    lam=2*lam1-lam0;
    alpha=(x11-x00)'/ds;
    bhta=(lam1-lam0)/ds;
    Nfun=@(x,lam) alpha*(x-x11)+bhta*(lam-lam1)-ds; %arc-length condition
    augfun=@(xlam) [eqfun(xlam(1:np,1),xlam(np+1,1));Nfun(xlam(1:np,1),xlam(np+1,1))];
    xlam0=[x;lam];
    if mod(k,every_k)==1
        [xlam,err,iterNew,J_x_lam]=newton_method(augfun,xlam0,xtol,ftol,epFD,maxiterNew);
    else
        tic;
        xlam = nsoli(xlam0,augfun,xtolJFNKgmres,parms);
        toc
    end
    x=xlam(1:np,1);
    lam=xlam(np+1,1);
    save_x(:,k)=x;
    save_lam(k)=lam;
    if flag_eigs==1
    tic;
    save_eigs(:,k)=eigs(@(v) stabfun(x,lam,v,delta),np,[],3,'largestabs','Tolerance',epFD);
    toc
    end
    lam0=lam1;
    lam1=lam;
    x00=x11;
    x11=x;
    fprintf('ARCLength ---- iter=%d, lambda=%2.4e \n', k,lam)
end
end