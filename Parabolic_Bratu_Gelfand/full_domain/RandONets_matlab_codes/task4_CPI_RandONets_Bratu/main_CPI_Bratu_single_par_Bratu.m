clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
rng(5)
load('RandONet_single_Bratu_10.mat')
%load('RandONet_parametric_Bratu_022_070.mat')
RandONet.C=gather(RandONet.C);
simmetry=0;
%load('')
%
x0=0; xf=1;
t0=0; 
Nx=51;
%Nt=501; %41;%31;
Nt=5;
nt=5;
dt=0.001; %0.05;
DT=0.015;
NNt=1+Nt*(nt+round(DT/dt))+nt;
tf=Nt*(nt*dt+DT)+nt*dt+t0; %tf=2; 40*0.05=2
xspan=linspace(x0,xf,Nx); %output grid
dx=xspan(2)-xspan(1);
tspan=linspace(t0,tf,NNt); %output times grid (define dt)
[XX,TT]=meshgrid(xspan,tspan);
%
%chebfun parameters
xcheb=chebfun('x',[x0,xf]);
bc.left=@(u) u; %diff(u,1);
bc.right=@(u) u; %diff(u,1);
optstrue=pdeset('RelTol',1e-6,'AbsTol',1e-8,'N',512);
%
lambda=1.0;
fcheb=@(t,x,u) diff(u,2)+lambda*exp(u);
%
%
%initial condition


l0=0.05*0.1;
u0_f=@(x) 4*l0*x.*(1-x);
u0cheb=u0_f(xcheb);
tic;
uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
true=uuchebtrue(xspan')';
toc
%truemax=sqrt(sum(abs(true).^2,2));
truemax=max(abs(true),[],2);
truemax_x=max(abs(diff(true,1,2)),[],2)/dx;
truemax_2x=max(abs(diff(true,2,2)),[],2)/dx.^2;

%
u0=true(1,:)'; %u0cheb(xspan');
fun_RON=@(x) EVAL_flags_RandONet(RandONet,[x;lambda],xspan',simmetry,parametric,flag_single);
%uu=time_stepper(fun_RON,u0,Nt);
[ttCPI,uuCPI]=CPI(fun_RON,u0,nt,dt,Nt,DT);
uuCPI=uuCPI';
[XXtt,TTtt]=meshgrid(xspan,ttCPI);
%
[uuRON]=time_stepper(fun_RON,u0,NNt);
%
%uuCPImax=sqrt(sum(uuCPI.^2,2));
uuCPImax=max(abs(uuCPI),[],2);
uuCPImax_x=max(abs(diff(uuCPI,1,2)),[],2)/dx;
uuCPImax_2x=max(abs(diff(uuCPI,2,2)),[],2)/dx^2;
%
uuRONmax=max(abs(uuRON),[],1);
uuRONmax_x=max(abs(diff(uuRON,1,1)),[],1)/dx;
uuRONmax_2x=max(abs(diff(uuRON,2,1)),[],1)/dx^2;
%
uchebtrue_tt=pde15s(fcheb,ttCPI,u0cheb,bc,optstrue);
true_tt=uchebtrue_tt(xspan')';
%

%FD
lam0=1.0;
%NxFD=51; %grid for Finite difference
xspanFD=xspan; %linspace(x0,xf,NxFD)';
%u00FD=spline(xspan,u0,xspanFD);
dxFD=xspanFD(2)-xspanFD(1);
dtFD=dxFD^2/2/lam0/2;
fun_FD=@(x) Bratu_FD_forwardEuler(x,lam0,dxFD,dtFD);
tmsp_FD=@(x) time_stepper(@(x) fun_FD(x),x,min(round(dt/dtFD)+1,101),1,Nx);
%eqsFD=@(x) x-tmsp_FD(x);



%
figure(1)
plot(xspan,true_tt(1,:),'k')
hold on
plot(xspan,uuCPI(1,:),'--r')
II=[1:NNt];
for i=II
plot(xspan,true(i,:),'k','HandleVisibility','off')
end
for i=1:length(ttCPI)
    plot(xspan,uuCPI(i,:),'--r','HandleVisibility','off')
end
legend('reference','PI-RandONet','Location','northwest')
%ylim([-1.1,1.1])
xlabel('$x$','Interpreter','latex')
ylabel('$u(x)$','Interpreter','latex')
set(gca,'FontSize',18)
grid on
pause(0.001)
%
folder='figures/';
equal_name='fig_Bratu_single_par_CPI';
type=1;
typename={'_rand'};
%
name='RandONet_vs_ref';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%
figure(2)
hold off
for i=1:Nt
    I=(1:nt+1)+(nt+1)*(i-1);
    iend=(nt+1)*(i-1)+nt+1;
    i0p=(nt+1)*(i)+1;
    surf(XXtt(I,:),TTtt(I,:),uuCPI(I,:),'LineStyle','none','FaceColor','interp')
    hold on
    v12=uuCPI(I(end),26);
    if i<Nt
        quiver3(0.5,ttCPI(iend),uuCPI(iend,26),...
                0,DT,uuCPI(i0p,26)-uuCPI(iend,26),...
            'LineWidth',2, 'MaxHeadSize', 5,...
            'Color',[0.5,0.5,0.5],'LineStyle','-','AutoScaleFactor',1)
    end
end
%view(2)
colorbar()
clim([0,0.09])
hold on

set(gca,'FontSize',18)
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
%
%
name='RandONet_space_time_traj';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%
figure(3)
hold off
surf(XX,TT,true,'LineStyle','none','FaceColor','interp')    
view(2)
colorbar()
clim([0,0.11])
set(gca,'FontSize',18)
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
%
name='ref_space_time_traj';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%
figure(4)
hold off
for i=1:Nt
    I=(1:nt+1)+(nt+1)*(i-1);
    contourf(XXtt(I,:),TTtt(I,:),abs(true_tt(I,:)-uuCPI(I,:))+1e-15,10.^(-16:1:16),'LineWidth',0.01,'ShowText','off','LineColor',[.3 .3 .3],'LineStyle','none')
    hold on
end
set(gca,'ColorScale','log')
colorbar()
c = colorbar;
colormap(jet)
ax = gca;
c_ax=clim(ax);
clim(ax,[max(c_ax(1),1e-4),max(1,c_ax(2))]);
c.Ticks = 10.^(-16:1:16);
c.Label.String = 'abs error';
c.Label.Interpreter = 'latex';
c.Label.FontSize = 18;
%
ax.FontName = 'Times';
ax.FontSize = 18;
%ax.LabelFontSizeMultiplier = 15/12;
xlabel('$x$','FontSize',18,'Interpreter','latex');
ylabel('$t$','FontSize',18,'Interpreter','latex','Rotation',0);
grid on
%
name='RandONet_space_time_error';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%

figure(5)
hold off
plot(tspan,truemax,'-b')
hold on
plot(tspan,uuRONmax,'--m')
plot(ttCPI,uuCPImax,'or','markersize',5)
legend('reference $u(x)$', ...
    'RandONet $u(x)$', ...
    'PI-RandONet $u(x)$', ...
    'interpreter','latex','Location','southeast')
set(gca, 'FontSize', 20)
xlabel('$t$', 'Interpreter', 'latex')
ylabel('$L^{\infty}$-norm', 'Interpreter', 'latex')
grid on
%
%
name='RandONet_CPI_relax_u';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%


figure(6)
hold off
plot(tspan,truemax_x,'-b')
hold on
plot(tspan,uuRONmax_x,'--m')
plot(ttCPI,uuCPImax_x,'or','markersize',5)
legend('reference $u_x(x)$', ...
    'RandONet $u_{x}(x)$', ...
    'PI-RandONet $u_x(x)$ ', ...
    'interpreter','latex','Location','southeast')
set(gca, 'FontSize', 20)
xlabel('$t$', 'Interpreter', 'latex')
ylabel('$L^{\infty}$-norm', 'Interpreter', 'latex')
%
grid on
name='RandONet_CPI_relax_u_x';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%
figure(7)
hold off
plot(tspan,truemax_2x,'-b')
hold on
plot(tspan,uuRONmax_2x,'--m')
plot(ttCPI,uuCPImax_2x,'or','markersize',5)
legend('reference $u_{xx}(x)$', ...
    'RandONet $u_{xx}(x)$', ...
    'PI-RandONet $u_{xx}(x)$', ...
    'interpreter','latex','Location','southeast')
set(gca, 'FontSize', 20)
grid on
xlabel('$t$', 'Interpreter', 'latex')
ylabel('$L^{\infty}$-norm', 'Interpreter', 'latex')
%
name='RandONet_CPI_relax_u_2x';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%
figure(8)
hold off
scatter3(XXtt(:), TTtt(:), uuCPI(:), 20, uuCPI(:), 'filled')  
set(gca, 'FontSize', 20)  
xlabel('$x$', 'Interpreter', 'latex')  
ylabel('$t$', 'Interpreter', 'latex')  
zlabel('$u(x,t)$', 'Interpreter', 'latex')  
colormap('parula');  
colorbar;  
view(3); % Ensure 3D view
%
name='RandONet_CPI_scatter3';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')

