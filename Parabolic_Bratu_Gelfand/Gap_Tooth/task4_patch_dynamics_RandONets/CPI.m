function [tt,uu]=CPI(fun,u0,nt,dt,Nt,DT)
uu=zeros(size(u0,1),1+Nt*(nt+1)+nt);
tt=zeros(1,1+Nt*(nt+1));
uu(:,1)=u0;
k=1;
for i=1:Nt
    for j=1:nt
        k=k+1;
        uu(:,k)=fun(uu(:,k-1));
        tt(k)=tt(k-1)+dt;
    end
    du=(uu(:,k)-uu(:,k-1))/dt;
    k=k+1;
    uu(:,k)=uu(:,k-1)+DT*du; %long
    tt(k)=tt(k-1)+DT;
end
%
for j=1:nt
    k=k+1;
    uu(:,k)=fun(uu(:,k-1));
    tt(k)=tt(k-1)+dt;
end

end