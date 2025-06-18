function [c0,c1,c_1,r0,r1,r_1]=Coll_Rea(f1,f_1,f0,rel,a1,a0,eps1,w,dt)
dact=f0(:,1)+f_1(:,1)+f1(:,1);
din=f0(:,2)+f_1(:,2)+f1(:,2);
d=[dact, din];
%termini di collisione
for i=1:2
    c1(:,i)=-rel(i)*(f1(:,i)-w(1)*d(:,i));
    c0(:,i)=-rel(i)*(f0(:,i)-w(2)*d(:,i));
    c_1(:,i)=-rel(i)*(f_1(:,i)-w(3)*d(:,i));
end
%termine di reazione per attivatore
r1(:,1)=w(1)*dt*(dact-dact.^3-din);
r0(:,1)=w(2)*dt*(dact-dact.^3-din);
r_1(:,1)=w(3)*dt*(dact-dact.^3-din);
% termine di reazione per inibitore
r1(:,2)=w(1)*dt*eps1*(dact-a1*din-a0);
r0(:,2)=w(2)*dt*eps1*(dact-a1*din-a0);
r_1(:,2)=w(3)*dt*eps1*(dact-a1*din-a0);