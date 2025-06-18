function u1=AC_FD_forwardEuler(u0,epsilon,dx,dt)
    u_xx=(u0(1:end-2,:)-2*u0(2:end-1,:)+u0(3:end,:))/dx^2;
    u0int=u0(2:end-1);
    ul=(4*u0int(1,:)-u0int(2,:))/3;
    ur=(4*u0int(end,:)-u0int(end-1,:))/3;
    rhs0=(epsilon*u_xx+1/epsilon*(u0int-u0int.^3));
    u1int=u0int+dt*rhs0;
    u1=[ul;u1int;ur];
end