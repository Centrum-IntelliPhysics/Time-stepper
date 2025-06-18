%run bifurcation part
%run_bifurcation_part
Ntlong=21;
u00RON=time_stepper(@(x) fun_RON(x,ep0),u0RON,Ntlong);
u00FD=time_stepper(@(x) fun_FD(x,ep0),u0FD,(Ntlong-1)*dt/dtFD+1);

%
figure(1)
hold off
plot(xspan,u0RON,'k-')
hold on
plot(xspan,u00RON,'b-')
plot(xspanFD,u00FD,'g:')
pause(0.001)

if flag_RON_load==0
disp('##### RandONet arc-length ####')
%[save_xRON,save_epRON]=arc_length_cont(eqs_RON,u00RON,ep0,d_ep,ep_final,200,1e-4,1e-3,1e-3,10);
[save_xRON,save_epRON,save_eigsRON]=arc_length_cont_JFNKgmres(tmsp_RON,u00RON,ep0,d_ep,ep_final,200,1e-6,1e6,1e-6,40);

if save_epRON(end)<ep_final/2
    disp('something FAILED ###############')
    d_ep=d_ep*2;
    [save_xRON2,save_epRON2,save_eigsRON2]=arc_length_cont_JFNKgmres(tmsp_RON,save_xRON(:,end),save_epRON(end)*1.1,d_ep,ep_final,200,1e-5,1e-5,1e-2,40);
    save_xRON=[save_xRON,save_xRON2];
    save_epRON=[save_epRON,save_epRON2];
    save_eigsRON=[save_eigsRON,save_eigsRON2];
end
%
if save_epRON(end)<ep_final*2/3
    disp('something FAILED ###############')
    [save_xRON2,save_epRON2]=arc_length_cont_JFNKgmres(tmsp_RON,save_xRON(:,end),save_epRON(end)*1.1,d_ep,ep_final,200,1e-4,1e-4,1e-4,40);
    save_xRON=[save_xRON,save_xRON2];
    save_epRON=[save_epRON,save_epRON2];
    save_eigsRON=[save_eigsRON,save_eigsRON2];
end
end

if flag_FD_load==0
    disp('##### FD arc-length ####')
%effective arc_length_continuation
%[save_xFD,save_epFD]=arc_length_cont(eqsFD,u0FD,ep0,d_ep,ep_final,200,1e-8,1e-6,1e-6,10);
[save_xFD,save_epFD,save_eigsFD]=arc_length_cont_JFNKgmres(tmsp_FD,u0FD,ep0,d_ep,ep_final,200,1e-10,1e-10,1e-7,40);
end
%

%

%
%