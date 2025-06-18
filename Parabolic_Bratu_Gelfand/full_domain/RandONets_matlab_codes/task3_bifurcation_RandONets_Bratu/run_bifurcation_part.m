%run bifurcation part
%run_bifurcation_part
Ntlong=41;
u00RON=time_stepper(@(x) fun_RON(x,lam0),u0RON,Ntlong);
u00FD=time_stepper(@(x) fun_FD(x,lam0),u0FD,round((Ntlong-1)*dt/dtFD)+1);

%
figure(1)
hold off
plot(xspan,u0RON,'k-')
hold on
plot(xspan,u00RON,'b-')
plot(xspanFD,u00FD,'g--')
pause(0.001)

ntot=600;
disp('##### RandONet arc-length ####')
%[save_xRON,save_epRON]=arc_length_cont(eqs_RON,u00RON,lam0,d_lam,lam_final,200,1e-4,1e-3,1e-3,10);
[save_xRON,save_epRON,save_eigsRON]=arc_length_cont_JFNKgmres(tmsp_RON,u00RON,lam0,d_lam,lam_final,ntot,1e-6,1e6,1e-6,40);
%

if flag_FD_load==0
    disp('##### FD arc-length ####')
%effective arc_length_continuation
%[save_xFD,save_epFD]=arc_length_cont(eqsFD,u0FD,lam0,d_lam,lam_final,200,1e-8,1e-6,1e-6,10);
[save_xFD,save_epFD,save_eigsFD]=arc_length_cont_JFNKgmres(tmsp_FD,u0FD,lam0,d_lam,lam_final,ntot,1e-10,1e-10,1e-7,40);
end
%

%

%
%