function [x,errf,iter,J]=newton_method(f,x0,xtol,ftol,ep,maxiter)
if nargin<3
    xtol=1e-14;
end
if nargin<4
    ftol=1e-12;
end
if nargin<5
    ep=1e-8;
end
if nargin<6
    maxiter=100;
end
ff=f(x0);
n=max(size(ff));
errf=norm(ff);
errx=xtol*10;
iter=0;
x=x0;
flag=0;
while errf>ftol && errx>xtol && iter<maxiter
    flag=1;
    iter=iter+1;
    xp=x;
    xm=x;
    J=zeros(n,n);
    for i=1:n
        xp(i)=xp(i)+ep;
        xm(i)=xm(i)-ep;
        fp=f(xp); fm=f(xm);
        J(:,i)=(fp-fm)/(2*ep);
        xp(i)=x(i);
        xm(i)=x(i);
    end
    dx=-pinv(J,1e-8)*ff;
    x=x+dx;
    ff=f(x);
    errf=norm(ff);
    errx=norm(dx);
    fprintf('newton iter %d, err=%2.4e\n', iter, errf)
end
if flag==0
    xp=x;
    xm=x;
    J=zeros(n,n);
    for i=1:n
        xp(i)=xp(i)+ep;
        xm(i)=xm(i)-ep;
        fp=f(xp); fm=f(xm);
        J(:,i)=(fp-fm)/(2*ep);
        xp(i)=x(i);
        xm(i)=x(i);
    end
end
end