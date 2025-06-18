clear
clc
close all
%
set(0,'DefaultLineLineWidth',2)
%
load('RandONet_parametric_FHN_0005_0995.mat')
pod_modes=1;
if pod_modes==1
load('PODmodes.mat')
end
%
flag_save_FD=0;
if flag_save_FD==0
    load('BD_FD_FHN')
end
%FHN PDE
variation_on=0; %<---------------------
RandONet.C=gather(RandONet.C);
simmetry=0;
%load('')
%
x0=0; xL=20;
t0=0; 

%FHN parameters
% Setting parameters
Du = 1; %coefficient of diffusion activator
Dv = 4; %coefficient of diffusion inibitor
a0 = -0.03; %parameter of the problem
a1 = 2;

%
%RandONet
Nx=41; %grid of RandONet
dt=0.1; %time step of the RandONet
dtRON=0.1; %0.05;
xspan=linspace(x0,xL,Nx)'; %output grid
dx=xspan(2)-xspan(1);
simmetry=0;
variation_on=0;
if variation_on==1
fun_RON=@(x,ep) x+EVAL_flags_RandONet(RandONet,x,ep,xspan,parametric);
else
fun_RON=@(x,ep) EVAL_flags_RandONet(RandONet,Vkov_proj*x,ep,xspan,parametric);
end
eqs_RON=@(uv0,ep) uv0-fun_RON(uv0,ep);

%FD
NxFD=41; %grid for Finite difference
xspanFD=linspace(x0,xL,NxFD)';
dxFD=xspanFD(2)-xspanFD(1);
dtFD=0.001;%dxFD^2/2/0.7/2;
fun_FD=@(x,ep) FD_FHN(x,Du,Dv,a1,a0,ep,dtFD,Nx,dx);
tmsp_FD_FHN=@(x,ep) time_stepper_end(@(x) fun_FD(x,ep),x,min(round(dt/dtFD)+1,101));
eqsFD=@(x,ep) x-tmsp_FD_FHN(x,ep);%(@(x) fun_FD(x,ep),x,min(round(dt/dtFD)+1,100));

%first branch sin n=1
%branch 1
ep0=0.2;
%initial condition
nn=1;

%
w_u_ic=0.9; a_u_ic=0.8; c_u_ic=11; b_u_ic=0.1;
u0=w_u_ic*tanh(a_u_ic*(xspan-c_u_ic))+b_u_ic;
w_v_ic=0.15; a_v_ic=0.4; c_v_ic=5; b_v_ic=0.02;
v0=w_v_ic*tanh(a_v_ic*(xspan-c_v_ic))+b_v_ic;
uv0=[u0;v0];
%
d_ep=0.005;
ep_final=1;
itersbd=85;
run_bifurcation_part
save1_xRON=save_xRON;
save1_epRON=save_epRON;
save1_eigsRON=save_eigs_RON;
if flag_save_FD==1
save1_eigsFD=save_eigs_FD;
save1_xFD=save_xFD;
save1_epFD=save_epFD;
end
%
flag_save_FD=0;
d_ep=-0.005;
itersbd=105;
run_bifurcation_part
save2_xRON=save_xRON;
save2_epRON=save_epRON;
save2_eigsRON=save_eigs_RON;
if flag_save_FD==1
save2_eigsFD=save_eigs_FD;
save2_xFD=save_xFD;
save2_epFD=save_epFD;
end

figure(2)
hold off
plot(save1_epFD,mean(save1_xFD(1:Nx,:),1),'b-')
hold on
plot(save1_epRON,mean(save1_xRON(1:Nx,:),1),'r--')
plot(save2_epFD,mean(save2_xFD(1:Nx,:),1),'b-','HandleVisibility','off')
plot(save2_epRON,mean(save2_xRON(1:Nx,:),1),'r--','HandleVisibility','off')
legend('Euler FD','POD-RandONet')
set(gca,'FontSize',20,'XTick',-2:0.2:2)
xlabel('$\varepsilon$','Interpreter','latex')
ylabel('$<u>$','Interpreter','latex')
ylim([-1,0])
grid on
pause(0.001)
%
% Inset plot
axes('Position', [0.1, 0.1, 0.2, 0.2])  % [left, bottom, width, height]
plot(xspanFD, save1_xFD(1:41, 70), 'b')
hold on
plot(xspan, save1_xRON(1:41, 70), 'r--')
set(gca,'xtick',[],'ytick',-2.1:0.3:2,'FontSize',10)
%xlabel('$x$', 'Interpreter', 'latex')
%ylabel('$u(x)$', 'Interpreter', 'latex')

%arrow
annotation('textarrow', ...
    [save1_epFD(70) / max(save1_epFD), save1_epFD(70) / max(save1_epFD)], ... % Start and end x position of the arrow
    [max(save1_xFD(:,70)) / max(max(save1_xFD)), max(save1_xFD(:,70)) / max(max(save1_xFD))], ... % Start and end y position of the arrow
    'String', '$\varepsilon= 0.5$', 'Interpreter', 'latex', 'FontSize', 12)

% Inset plot
axes('Position', [0.4, 0.5, 0.2, 0.2])  % [left, bottom, width, height]
plot(xspanFD, save1_xFD(1:Nx, 1), 'b')
hold on
plot(xspan, save1_xRON(1:Nx, 1), 'r--')
set(gca,'xtick',[],'ytick',-2:0.4:2,'FontSize',10)
%xlabel('$x$', 'Interpreter', 'latex')
%ylabel('$u(x)$', 'Interpreter', 'latex')
%
annotation('textarrow', ...
    [save1_epFD(1) / max(save1_epFD), save1_epFD(1) / max(save1_epFD)], ... % Start and end x position of the arrow
    [max(save1_xFD(:,1)) / max(max(save1_xFD)), max(save1_xFD(:,1)) / max(max(save1_xFD))], ... % Start and end y position of the arrow
    'String', '$\varepsilon= 0.2$', 'Interpreter', 'latex', 'FontSize', 12)




inde=1:2;
ind=1:45;
figure(3)
hold off
plot(save1_epFD,abs(save1_eigsFD(inde,:)),'ob')
hold on
plot(save1_epRON,abs(save1_eigsRON(inde,:)),'*r','MarkerSize',4)

plot(save2_epFD,abs(save2_eigsFD(inde,:)),'ob','HandleVisibility','off')
hold on
plot(save2_epRON,abs(save2_eigsRON(inde,:)),'*r','MarkerSize',4,'HandleVisibility','off')
plot([0,1],[1,1],'--k','HandleVisibility','off')
legend('Euler FD','','POD-RandONet')
set(gca,'FontSize',20)
grid on
xlabel('$\varepsilon$','Interpreter','latex')
ylabel('eigenvalues $|\mu|$','Interpreter','latex')


theta=linspace(-pi/5,pi/5,1001);
figure(4)
hold off
plot(save2_eigsFD(1,:),'ob')
hold on
plot(save2_eigsFD(2,:),'ob','HandleVisibility','off')
plot(save2_eigsFD(3,:),'ob','HandleVisibility','off')
plot(save2_eigsRON,'*r','MarkerSize',4)
plot(cos(theta)+1i*sin(theta),'--k','HandleVisibility','off')
legend('Euler FD','','','POD-RandONet')
set(gca,'FontSize',20)
grid on
xlabel('$Re(\mu)$','Interpreter','latex')
ylabel('$Im(\mu)$','Interpreter','latex')
xlim([1-0.015,1+0.005])
ylim([-0.01,0.01])

fig5=figure(5)
subplot(1,2,1)
hold off
plot(save2_eigsFD(1,:),'ob')
hold on
plot(save2_eigsFD(2,:),'ob','HandleVisibility','off')
plot(save2_eigsFD(3,:),'ob','HandleVisibility','off')
plot(save2_eigsRON,'*r','MarkerSize',4)
plot(cos(theta)+1i*sin(theta),'--k','HandleVisibility','off')
legend('Euler FD','','','POD-RandONet')
set(gca,'FontSize',20)
grid on
xlabel('$Re(\mu)$','Interpreter','latex')
ylabel('$Im(\mu)$','Interpreter','latex')
xlim([1-0.015,1+0.005])
ylim([-0.01,0.01])
%
subplot(1,2,2)
hold off
plot(save1_epFD,abs(save1_eigsFD(inde,:)),'ob')
hold on
plot(save1_epRON,abs(save1_eigsRON(inde,:)),'*r','MarkerSize',4)
plot(save2_epFD,abs(save2_eigsFD(inde,:)),'ob','HandleVisibility','off')
hold on
plot(save2_epRON,abs(save2_eigsRON(inde,:)),'*r','MarkerSize',4,'HandleVisibility','off')
plot([0,1],[1,1],'--k','HandleVisibility','off')
legend('Euler FD','','POD-RandONet')
set(gca,'FontSize',20)
grid on
xlabel('$\varepsilon$','Interpreter','latex')
ylabel('eigenvalues $|\mu|$','Interpreter','latex')

fig5.Position(3)=1120;

if flag_save_FD==1
    save('BD_FD_FHN','save1_epFD','save1_eigsFD','save1_xFD',...
        'save2_xFD','save2_epFD','save2_eigsFD')
end