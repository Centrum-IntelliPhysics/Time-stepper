%run bifurcation part
%run_bifurcation_part
Nt=501; %41;%31;
uv00RON=time_stepper_end(@(x) fun_RON(x,ep0),uv0,Nt);
uv00FD=time_stepper_end(@(x) tmsp_FD_FHN(x,ep0),uv0,Nt);

%
figure(1)
hold off
plot(xspan,uv0(1:Nx,1),'k-')
hold on
plot(xspan,uv00RON(1:Nx,1),'b-')
plot(xspanFD,uv00FD(1:Nx,1),'g:')
pause(0.001)


%effective arc_length_continuation
%[save_xFD,save_epFD]=arc_length_cont(eqsFD,u0FD,ep0,d_ep,ep_final,200,1e-8,1e-6,1e-6,10);
%(fun,x0,lam0,d_lam,lamfinal,maxiters,xtol,ftol,epFD,maxiterNew)
if flag_save_FD==1
[save_xFD,save_epFD,save_eigs_FD]=arc_length_cont_JFNKgmres(tmsp_FD_FHN,uv00FD,ep0,d_ep,ep_final,itersbd,1e-8,1e-7,1e-6,40);
end
%[save_xFD,save_epFD]=arc_length_cont_JFNKgmres(tmsp_FD_FHN,uv00FD,ep0,d_ep,ep_final,200,1e-10,1e-8,1e-7,40);

%

%

%[save_xRON,save_epRON]=arc_length_cont(eqs_RON,u00RON,ep0,d_ep,ep_final,200,1e-4,1e-3,1e-3,10);
[save_xRON,save_epRON,save_eigs_RON]=arc_length_cont_JFNKgmres(fun_RON,uv00RON,ep0,d_ep,ep_final,itersbd,1e-6,1e-5,1e-5,40);
%
%