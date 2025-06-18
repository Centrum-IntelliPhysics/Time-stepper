clear
clc
close all
%
set(0,'DefaultLineLineWidth',2)
%
flag_FD_load=1;
if flag_FD_load==1
load('Bratu_FD_bif_diag.mat')
end
%
%Allen-Cahn PDE
x0=0; xf=1;
t0=0;

%
%RandONet
Nx=51; %grid of RandONet
Nt=3;
dt=0.001; %time step of the RandONet
xspan=linspace(x0,xf,Nx)'; %output grid
load('RandONet_parametric_Bratu_00_38.mat')
simmetry=0;
flag_single=0;
fun_RON=@(x,lam) EVAL_flags_RandONet(RandONet,[x;lam],xspan,simmetry,parametric,flag_single);
tmsp_RON=@(x,lam) time_stepper(@(x) fun_RON(x,lam),x,Nt);
eqs_RON=@(x,lam) x-tmsp_RON(x,lam);

%FD
NxFD=51; %grid for Finite difference
xspanFD=linspace(x0,xf,NxFD)';
dxFD=xspanFD(2)-xspanFD(1);
dtFD=dxFD^2/2/1/2;
fun_FD=@(x,lam) Bratu_FD_forwardEuler(x,lam,dxFD,dtFD);
tmsp_FD=@(x,lam) time_stepper(@(x) fun_FD(x,lam),x,round((Nt-1)*dt/dtFD)+1);
eqsFD=@(x,lam) x-tmsp_FD(x,lam);

%branch 1
lam0=0.1;
lam_final=3.8;
%initial condition
u0_f=@(x) 1*x.*(1-x); %sb*sin(((nn-1)*pi+pi/2)*x);
u0RON=u0_f(xspan);
u0FD=u0_f(xspanFD);
%
d_lam=0.03;
run_bifurcation_part
save1_xRON=save_xRON;
save1_epRON=save_epRON;
save1_eigsRON=save_eigsRON;
if flag_FD_load==0
save1_xFD=save_xFD;
save1_epFD=save_epFD;
save1_eigsFD=save_eigsFD;
end

figure(2)
plot(save1_epFD,max(save1_xFD,[],1),'k-')
hold on
plot(save1_epRON,max(save1_xRON,[],1),'r--')
legend('Euler FD','RandONet')
set(gca,'FontSize',18)
xlabel('parameter $\lambda$','Interpreter','latex')
ylabel('$\|u\|_{\infty}$','Interpreter','latex')
pause(0.001)
%
% Inset plot
axes('Position', [0.1, 0.1, 0.2, 0.2])  % [left, bottom, width, height]
plot(xspanFD, save1_xFD(:, 32), 'b')
hold on
plot(xspan, save1_xRON(:, 32), 'r--')
set(gca,'xtick',[],'ytick',0:0.05:0.15,'FontSize',10)
%xlabel('$x$', 'Interpreter', 'latex')
%ylabel('$u(x)$', 'Interpreter', 'latex')

%arrow
annotation('textarrow', ...
    [save1_epFD(32) / max(save1_epFD), save1_epFD(32) / max(save1_epFD)], ... % Start and end x position of the arrow
    [max(save1_xFD(:,32)) / max(max(save1_xFD)), max(save1_xFD(:,32)) / max(max(save1_xFD))], ... % Start and end y position of the arrow
    'String', '$\lambda= 1$', 'Interpreter', 'latex', 'FontSize', 12)

% Inset plot
axes('Position', [0.4, 0.5, 0.2, 0.2])  % [left, bottom, width, height]
plot(xspanFD, save1_xFD(:, 429), 'b')
hold on
plot(xspan, save1_xRON(:, 429), 'r--')
set(gca,'xtick',[],'ytick',0:1:3,'FontSize',10)
%xlabel('$x$', 'Interpreter', 'latex')
%ylabel('$u(x)$', 'Interpreter', 'latex')
%
annotation('textarrow', ...
    [save1_epFD(32) / max(save1_epFD), save1_epFD(32) / max(save1_epFD)], ... % Start and end x position of the arrow
    [max(save1_xFD(:,32)) / max(max(save1_xFD)), max(save1_xFD(:,32)) / max(max(save1_xFD))], ... % Start and end y position of the arrow
    'String', '$\lambda= 2$', 'Interpreter', 'latex', 'FontSize', 12)


figure(3)
hold off
plot(save1_epFD,abs(save1_eigsFD),'-b')
hold on
plot(save1_epRON,abs(save1_eigsRON),'--r')
plot([0,4],[1,1],'--k','HandleVisibility','off')
legend('Euler FD','','','RandONet')
set(gca,'FontSize',18)
grid on
xlabel('parameter $\lambda$','Interpreter','latex')
ylabel('eigenvalues $|\mu|$','Interpreter','latex')
pause(0.001)


if flag_FD_load==0
save('Bratu_FD_bif_diag.mat','save1_epFD','save1_xFD','save1_eigsFD')
end

