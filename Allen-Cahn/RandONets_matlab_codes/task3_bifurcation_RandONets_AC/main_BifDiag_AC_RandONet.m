clear
clc
close all
%
set(0,'DefaultLineLineWidth',2)
%
flag_FD_load=1;
if flag_FD_load==1
load('AC_FD_bif_diag.mat')
end
flag_RON_load=1;
if flag_RON_load==1
load('AC_RON_bif_diag.mat')
end
%
%Allen-Cahn PDE
x0=-1; xf=1;
t0=0;

%
%RandONet
Nx=100; %grid of RandONet
dt=0.01; %time step of the RandONet
xspan=linspace(x0,xf,Nx)'; %output grid
Nt=3;
load('RandONet_parametric_AC_022_070.mat')
simmetry=1;
fun_RON=@(x,ep) EVAL_flags_RandONet(RandONet,[x;ep],xspan,simmetry,parametric,flag_single);
tmsp_RON=@(x,ep) time_stepper(@(x) fun_RON(x,ep),x,Nt);
eqs_RON=@(x,ep) x-tmsp_RON(x,ep);

%FD
NxFD=101; %grid for Finite difference
xspanFD=linspace(x0,xf,NxFD)';
dxFD=xspanFD(2)-xspanFD(1);
dtFD=dxFD^2/2/1;
fun_FD=@(x,ep) AC_FD_forwardEuler(x,ep,dxFD,dtFD);
tmsp_FD=@(x,ep) time_stepper(@(x) fun_FD(x,ep),x,(Nt-1)*dt/dtFD+1);
eqsFD=@(x,ep) x-tmsp_FD(x,ep);

%first branch sin n=1
%branch 1
ep0=0.22;
%initial condition
nn=1;
epss_bif_sin=1./(pi/2+(nn-1)*pi);
ep_final=epss_bif_sin*1.01;
sb=1.2*sqrt(abs(ep0-epss_bif_sin));
sb=sb*(1+(2*rand(1)-1)/100);
u0_f=@(x) sb*sin(((nn-1)*pi+pi/2)*x);
u0RON=u0_f(xspan);
u0FD=u0_f(xspanFD);
%
d_ep=0.003;
%Nt=101; %41;%31;
run_bifurcation_part
if flag_RON_load==0
save1_xRON=save_xRON;
save1_epRON=save_epRON;
save1_eigsRON=save_eigsRON;
end
if flag_FD_load==0
save1_xFD=save_xFD;
save1_epFD=save_epFD;
save1_eigsFD=save_eigsFD;
end

fig2=figure(2); %sin
hold off
plot(save1_epFD,save1_xFD(1,:),'b-')
hold on
plot(save1_epFD,-save1_xFD(1,:),'b-','HandleVisibility','off')
plot(save1_epRON,save1_xRON(1,:),'r--')
plot(save1_epRON,-save1_xRON(1,:),'r--','HandleVisibility','off')
legend('Euler FD','RandONet')
set(gca,'FontSize',18)
xlabel('$\varepsilon$','Interpreter','latex')
ylabel('$\phi(-\!1)$','Interpreter','latex')
pause(0.001)
%
annotation(fig2,'textbox',...
    [0.755642857142857 0.713809523809525 0.0504285714285714 0.0457142857142868],...
    'String',{'(b)'},...
    'LineStyle','none',...
    'FontSize',18,...
    'FitBoxToText','off');


%second branch cos n=1
%branch 2
ep0=0.22;
%initial condition
nn=1;
epss_bif_cos=1./((nn)*pi);
ep_final=epss_bif_cos*1.02;
sb=1.1*sqrt(abs(ep0-epss_bif_sin));
sb=sb*(1+(2*rand(1)-1)/10);
u0_f=@(x) sb*cos(((nn)*pi)*x);
u0RON=u0_f(xspan);
u0FD=u0_f(xspanFD);
%
d_ep=0.002;
Nt=101; %41;%31;
run_bifurcation_part
if flag_RON_load==0
save2_xRON=save_xRON;
save2_epRON=save_epRON;
save2_eigsRON=save_eigsRON;
end
if flag_FD_load==0
save2_xFD=save_xFD;
save2_epFD=save_epFD;
save2_eigsFD=save_eigsFD;
end
%
figure(2) %cos
plot(save2_epFD,save2_xFD(1,:),'b-','HandleVisibility','off')
hold on
plot(save2_epFD,-save2_xFD(1,:),'b-','HandleVisibility','off')
plot(save2_epRON,save2_xRON(1,:),'r--','HandleVisibility','off')
hold on
plot(save2_epRON,-save2_xRON(1,:),'r--','HandleVisibility','off')
legend('Euler FD','RandONet')
pause(0.001)
annotation(fig2,'textbox',...
    [0.2935 0.741428571428574 0.0504285714285713 0.0457142857142868],...
    'String','(c)',...
    'LineStyle','none',...
    'FontSize',18,...
    'FitBoxToText','off');
%

%
%unstable branch u=0
%branch 3
ep0=0.22;
ep_final=epss_bif_sin*1.02;
%initial condition
u0_f=@(x) 0*x;
u0RON=u0_f(xspan);
u0FD=u0_f(xspanFD);
%
d_ep=0.005;
Nt=5; %41;%31;
run_bifurcation_part
if flag_RON_load==0
save3_xRON=save_xRON;
save3_epRON=save_epRON;
save3_eigsRON=save_eigsRON;
end
if flag_FD_load==0
save3_xFD=save_xFD;
save3_epFD=save_epFD;
save3_eigsFD=save_eigsFD;
end
%
figure(2) %zero-flat
plot(save3_epFD,save3_xFD(1,:),'b-','HandleVisibility','off')
hold on
plot(save3_epRON,save3_xRON(1,:),'r--','HandleVisibility','off')
legend('Euler FD','RandONet')
pause(0.001)
%
annotation(fig2,'textbox',...
    [0.436357142857142 0.602380952380955 0.0504285714285712 0.0457142857142866],...
    'String','(d)',...
    'LineStyle','none',...
    'FontSize',18,...
    'FitBoxToText','off');

figure(3)
hold off
plot(save1_epFD(:,1:2:end),abs(save1_eigsFD(:,1:2:end)),'ob')
hold on
plot(save1_epRON(:,1:2:end),abs(save1_eigsRON(:,1:2:end)),'*r','MarkerSize',4)
plot(save1_epFD,ones(size(save1_eigsFD)),'--k','HandleVisibility','off')
legend('Euler FD','','','RandONet')
set(gca,'FontSize',18)
grid on
xlabel('$\varepsilon$','Interpreter','latex')
ylabel('eigenvalues $|\mu|$','Interpreter','latex')
pause(0.001)

figure(4)
hold off
plot(save2_epFD(:,1:2:end),abs(save2_eigsFD(:,1:2:end)),'ob')
hold on
plot(save2_epRON(:,1:2:end),abs(save2_eigsRON(:,1:2:end)),'*r','MarkerSize',4)
plot(save2_epFD,ones(size(save2_eigsFD)),'--k','HandleVisibility','off')
legend('Euler FD','','','RandONet')
set(gca,'FontSize',18)
grid on
xlabel('$\varepsilon$','Interpreter','latex')
ylabel('eigenvalues $|\mu|$','Interpreter','latex')
pause(0.001)

figure(5)
hold off
plot(save3_epFD(:,1:2:end),abs(save3_eigsFD(:,1:2:end)),'ob')
hold on
plot(save3_epRON(:,1:2:end),abs(save3_eigsRON(:,1:2:end)),'*r','MarkerSize',4)
plot(save3_epFD,ones(size(save3_eigsFD)),'--k','HandleVisibility','off')
legend('Euler FD','','','RandONet')
set(gca,'FontSize',18)
grid on
xlabel('$\varepsilon$','Interpreter','latex')
ylabel('eigenvalues $|\mu|$','Interpreter','latex')
pause(0.001)

figure(2)
xlim([0.22,0.8])

%%%%
figure(2) %sin box
% Inset plot
axes('Position', [0.6, 0.1, 0.2, 0.2])  % [left, bottom, width, height]
plot(xspanFD, save1_xFD(:, 38), 'b')
hold on
plot(xspan, save1_xRON(:, 53), 'r--')
set(gca,'xtick',[],'ytick',-2:0.4:2,'FontSize',10)
%xlabel('$x$', 'Interpreter', 'latex')
%ylabel('$u(x)$', 'Interpreter', 'latex')
%arrow
annotation(fig2,'textarrow',[0.674642857142857 0.510357142857139],...
    [0.336666666666667 0.270000000000003],'String','$\varepsilon= 0.5$',...
    'Interpreter','latex',...
    'FontSize',12);%cos box
% Inset plot
axes('Position', [0.1, 0.1, 0.2, 0.2])  % [left, bottom, width, height]
plot(xspanFD, save2_xFD(:, 18), 'b')
hold on
plot(xspan, save2_xRON(:, 18), 'r--')
set(gca,'xtick',[],'ytick',-2:0.4:2,'FontSize',10)
%xlabel('$x$', 'Interpreter', 'latex')
%ylabel('$u(x)$', 'Interpreter', 'latex')

annotation(fig2,'textarrow',[0.360357142857143 0.184642857142857],...
    [0.304285714285714 0.268095238095238],'String','$\varepsilon= 0.25$',...
    'Interpreter','latex',...
    'FontSize',12);


if flag_FD_load==0
save('AC_FD_bif_diag.mat','save1_epFD','save1_xFD','save2_epFD','save2_xFD',...
    'save3_epFD','save3_xFD','save1_eigsFD','save2_eigsFD','save3_eigsFD')
end

if flag_RON_load==0
save('AC_RON_bif_diag.mat','save1_epRON','save1_xRON','save2_epRON','save2_xRON',...
    'save3_epRON','save3_xRON','save1_eigsRON','save2_eigsRON','save3_eigsRON')
end