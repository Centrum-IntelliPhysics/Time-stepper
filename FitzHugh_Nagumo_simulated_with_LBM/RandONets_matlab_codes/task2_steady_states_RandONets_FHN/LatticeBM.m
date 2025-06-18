function [f1,f_1,f0]=LatticeBM(f1,f_1,f0,rel,a1,a0,eps1,w,dt)
N=size(f1,1)-1;
%inizializzazione streaming
f1star=zeros(N+1,2);
f0star=zeros(N+1,2);
f_1star=zeros(N+1,2);

[c0,c1,c_1,r0,r1,r_1]=Coll_Rea(f1,f_1,f0,rel,a1,a0,eps1,w,dt);
%streaming
for j=1:2
    for i=1:N+1
         f1star(i,j)=f1(i,j)+c1(i,j)+r1(i,j);
         f0star(i,j)=f0(i,j)+c0(i,j)+r0(i,j);
         f_1star(i,j)=f_1(i,j)+c_1(i,j)+r_1(i,j);
    end
    f0(:,j)=f0star(:,j);
    for i=1:N
        f1(i+1,j)=f1star(i,j);
    end   
    f1(1,j)=f_1(1,j)+c_1(1,j)+r_1(1,j);
    for i=2:N+1
          f_1(i-1,j)=f_1star(i,j);
    end 
    f_1(N+1,j)=f1(N+1,j);
end    