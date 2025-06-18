function uv1=FD_FHN(uv0,Du,Dv,a1,a0,eps1,dt,Nx,dx)
u0=uv0(1:Nx,1);
v0=uv0(Nx+1:end,1);
%
u0_xx=(u0(3:end,1)-2*u0(2:end-1,1)+u0(1:end-2,1))/(dx^2);
v0_xx=(v0(3:end,1)-2*v0(2:end-1,1)+v0(1:end-2,1))/(dx^2);
%
rhsu=Du*u0_xx+u0(2:end-1,1)-u0(2:end-1,1).^3-v0(2:end-1,1);
%
rhsv=Dv*v0_xx+eps1*(u0(2:end-1)-a1*v0(2:end-1)-a0);
%
u1=u0(2:end-1,1)+dt*rhsu;
u1=[(4*u1(1,1)-u1(2,1))/3;u1;(4*u1(end,1)-u1(end-1,1))/3];
%
v1=v0(2:end-1,1)+dt*rhsv;
v1=[(4*v1(1,1)-v1(2,1))/3;v1;(4*v1(end,1)-v1(end-1,1))/3];
uv1=[u1;v1];
end