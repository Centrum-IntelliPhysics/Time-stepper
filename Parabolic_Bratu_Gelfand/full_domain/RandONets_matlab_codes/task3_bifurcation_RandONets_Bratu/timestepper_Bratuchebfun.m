function u=timestepper_Bratuchebfun(u0,fcheb,tspan,bc,optstrue,xspan)
u0cheb=chebfun(u0,'equi');
uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
uu=uuchebtrue(xspan);
u=uu(:,end);
end

