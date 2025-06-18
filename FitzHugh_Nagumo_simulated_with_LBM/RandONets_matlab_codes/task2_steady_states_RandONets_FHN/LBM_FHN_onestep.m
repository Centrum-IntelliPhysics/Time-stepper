function uv1=LBM_FHN_onestep(uv0,Nx,rel,a1,a0,eps1,w,dtLBM)
%NON FUNZIONA COSI
%BISOGNA LASCIARE EVOLVERE I STATI delle distribuzioni
%non si può resettare ad ogni time-step con f1=u*w1; f0=u*w2

u0=uv0(1:Nx,1);
v0=uv0(Nx+1:end,1);
%
f1=u0*w(1); %f distribution of activator
f0=u0*w(2);
f_1=u0*w(3);
%
f1(:,2)=w(1)*v0; %f distribution of inibitor
f0(:,2)=w(2)*v0;
f_1(:,2)=w(3)*v0;
%
[f1,f_1,f0]=LatticeBM(f1,f_1,f0,rel,a1,a0,eps1,w,dtLBM);
%
u1=f0(:,1)+f_1(:,1)+f1(:,1);
v1=f0(:,2)+f_1(:,2)+f1(:,2);
uv1=[u1;v1];
end