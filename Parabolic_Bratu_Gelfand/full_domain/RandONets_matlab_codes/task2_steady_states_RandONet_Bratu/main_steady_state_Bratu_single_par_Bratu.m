clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
load('RandONet_single_Bratu_10.mat')
%load('RandONet_parametric_Bratu_022_070.mat')
RandONet.C=gather(RandONet.C);
simmetry=0;
%load('')
%
x0=0; xf=1;
t0=0; 
Nx=51;
Nt=501; %41;%31;
dt=0.001; %0.05;
tf=(Nt-1)*dt+t0; %tf=2; 40*0.05=2
xspan=linspace(x0,xf,Nx); %output grid
tspan=linspace(t0,tf,Nt); %output times grid (define dt)
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

%
u0=true(1,:)'; %u0cheb(xspan');
fun_RON=@(x) EVAL_flags_RandONet(RandONet,[x;lambda],xspan',simmetry,parametric,flag_single);
uu=time_stepper(fun_RON,u0,Nt);
uu=uu';
toc
%
figure(1)
plot(xspan,true(1,:),'k')
hold on
plot(xspan,uu(1,:),'--r')
II=[2,5,10:10:100,150,Nt];
for i=II
plot(xspan,true(i,:),'k','HandleVisibility','off')
plot(xspan,uu(i,:),'--r','HandleVisibility','off')
end
legend('reference','RandONet','Location','northwest')
%ylim([-1.1,1.1])
xlabel('$x$','Interpreter','latex')
ylabel('$u(x)$','Interpreter','latex')
set(gca,'FontSize',18)
grid on
pause(0.001)
%
folder='figures/';
equal_name='fig_Bratu_single_par_';
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
surf(XX,TT,uu,'LineStyle','none','FaceColor','interp')
view(2)
colorbar()
clim([0,0.2])
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
surf(XX,TT,true,'LineStyle','none','FaceColor','interp')
view(2)
colorbar()
clim([0,0.2])
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
contourf(XX,TT,abs(true-uu)+1e-15,10.^(-16:1:16),'LineWidth',0.01,'ShowText','off','LineColor',[.3 .3 .3],'LineStyle','none')
set(gca,'ColorScale','log')
colorbar()
c = colorbar;
colormap(jet)
ax = gca;
c_ax=clim(ax);
clim(ax,[max(c_ax(1),1e-5),max(1,c_ax(2))]);
c.Ticks = 10.^(-16:1:16);
c.Label.String = 'abs error';
c.Label.Interpreter = 'latex';
c.Label.FontSize = 16;
%
ax.FontName = 'Times';
ax.FontSize = 16;
%ax.LabelFontSizeMultiplier = 15/12;
xlabel('$x$','FontSize',18,'Interpreter','latex');
ylabel('$t$','FontSize',18,'Interpreter','latex','Rotation',0);
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
contourf(XX(1:end-1,:),TT(1:end-1,:),abs(uu(2:end,:)-uu(1:end-1,:)),...
    10.^(-16:1:16),'LineWidth',0.01,'ShowText','off','LineColor',[.3 .3 .3],'LineStyle','none')
set(gca,'ColorScale','log')
colorbar()
c = colorbar;
colormap(jet)
ax = gca;
c_ax=clim(ax);
clim(ax,[c_ax(1),max(1,c_ax(2))]);
c.Ticks = 10.^(-16:1:16);
c.Label.String = "$u'(t)$";
c.Label.Interpreter = 'latex';
c.Label.FontSize = 16;
%
ax.FontName = 'Times';
ax.FontSize = 16;
%ax.LabelFontSizeMultiplier = 15/12;
xlabel('$x$','FontSize',18,'Interpreter','latex');
ylabel('$t$','FontSize',18,'Interpreter','latex','Rotation',0);
%
figure(6)
hold off
contourf(XX(1:end-1,:),TT(1:end-1,:),abs(true(2:end,:)-true(1:end-1,:)),...
    10.^(-16:1:16),'LineWidth',0.01,'ShowText','off','LineColor',[.3 .3 .3],'LineStyle','none')
set(gca,'ColorScale','log')
colorbar()
c = colorbar;
colormap(jet)
ax = gca;
c_ax=clim(ax);
clim(ax,[c_ax(1),max(1,c_ax(2))]);
c.Ticks = 10.^(-16:1:16);
c.Label.String = "$u'(t)$";
c.Label.Interpreter = 'latex';
c.Label.FontSize = 16;
%
ax.FontName = 'Times';
ax.FontSize = 16;
%ax.LabelFontSizeMultiplier = 15/12;
xlabel('$x$','FontSize',18,'Interpreter','latex');
ylabel('$t$','FontSize',18,'Interpreter','latex','Rotation',0);


%%%%%%%%%%%%%%%%%
u00RON=uu(end,:)';
tmsp_RON=@(x) time_stepper(@(x) fun_RON(x),x,2,1);
eqs_RON=@(x) x-tmsp_RON(x);

%FD
lam0=1.0;
NxFD=51; %grid for Finite difference
xspanFD=linspace(x0,xf,NxFD)';
u00FD=spline(xspan,u00RON,xspanFD);
dxFD=xspanFD(2)-xspanFD(1);
dtFD=dxFD^2/2/lam0/2;
fun_FD=@(x) Bratu_FD_forwardEuler(x,lam0,dxFD,dtFD);
tmsp_FD=@(x) time_stepper(@(x) fun_FD(x),x,min(round(dt/dtFD)+1,101),1,Nx);
eqsFD=@(x) x-tmsp_FD(x);


%%%%%%steady-states
[u00nRON,errf,iter,Jron]=newton_method(@(x) eqs_RON(x),u00RON,1e-6,1e-6,1e-6,100);
%
[u00nFD,errfFD,iterFD,Jfd]=newton_method(@(x) eqsFD(x),u00FD,1e-12,1e-10,1e-7,100);
%

figure(7)
hold off
plot(xspanFD,u00nFD,'b-')
hold on
plot(xspan,u00nRON,'r--')
legend('Euler FD','RandONet','Location','northwest')
set(gca,'FontSize',18)
xlabel('$x$','Interpreter','latex')
ylabel('$u(x)$','Interpreter','latex')
grid on
%
name='RandONet_steady_states';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')


%%%%%eigenvalues
Jron=eye(Nx)-Jron;
[Vron,Eron]=eig(Jron);
Eron=diag(Eron);
%
Jfd=eye(NxFD)-Jfd;
[Vfd,Efd]=eig(Jfd);
Efd=diag(Efd);
%
theta=linspace(0,2*pi,1001);
%
figure(8)
hold off
plot(cos(theta)+1j*sin(theta),'--k','HandleVisibility','off')
hold on
plot(Efd,'*b')
plot(Eron,'or')
legend('Euler FD','RandONet')
set(gca,'FontSize',18)
xlabel('$Re(\mu$)','Interpreter','latex')
ylabel('$Im(\mu$)','Interpreter','latex')
grid on
xlim([-1.1,1.1])
ylim([-1.1,1.1])
%
name='RandONet_full_spectrum';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')

%dominant eigenvalues with Jacobian-Free
epfd=1e-6;
Jvron=@(x) (tmsp_RON(u00nRON+epfd*x)-tmsp_RON(u00nRON))/epfd;
Jvfd=@(x) (tmsp_FD(u00nFD+epfd*x)-tmsp_FD(u00nFD))/epfd;
[Vron_gm,Eron_gm]=eigs(Jvron,Nx,[],3,'largestabs','Tolerance',1e-6);
[Vfd_gm,Efd_gm]=eigs(Jvfd,NxFD,[],3,'largestabs','Tolerance',1e-8);
Eron_gm=diag(Eron_gm);
Efd_gm=diag(Efd_gm);
figure(9)
hold off
plot(cos(theta),sin(theta),'--k','HandleVisibility','off')
hold on
plot(real(Efd_gm),imag(Efd_gm),'*b')
plot(real(Eron_gm),imag(Eron_gm),'or')
legend('Euler FD','RandONet')
set(gca,'FontSize',18)
xlabel('$Re(\mu$)','Interpreter','latex')
ylabel('$Im(\mu$)','Interpreter','latex')
grid on
%
xm=mean([real(Efd_gm);real(Eron_gm)]);
ylen=max([imag(Efd_gm);imag(Eron_gm)])*1.1;
xstd=std([real(Efd_gm);real(Eron_gm)]);
dxdx=max(3*xstd,ylen);
xlim([xm-3*xstd,xm+3*xstd])
ylim([-dxdx,dxdx])
%
name='RandONet_3_leading_eigs';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')


%%%

figure(10)
subplot(1,2,1)
hold off
plot(cos(theta)+1j*sin(theta),'--k','HandleVisibility','off')
hold on
plot(Efd,'*b')
plot(Eron,'or')
legend('Euler FD','RandONet')
set(gca,'FontSize',18)
xlabel('$Re(\mu$)','Interpreter','latex')
ylabel('$Im(\mu$)','Interpreter','latex')
grid on
xlim([-1.1,1.1])
ylim([-1.1,1.1])
%%%%%%%
subplot(1,2,2)
hold off
plot(cos(theta),sin(theta),'--k','HandleVisibility','off')
hold on
plot(real(Efd_gm),imag(Efd_gm),'*b','MarkerSize',10)
plot(real(Eron_gm),imag(Eron_gm),'or','MarkerSize',10)
legend('Euler FD','RandONet')
set(gca,'FontSize',18)
xlabel('$Re(\mu$)','Interpreter','latex')
ylabel('$Im(\mu$)','Interpreter','latex')
grid on
%
xm=mean([real(Efd_gm);real(Eron_gm)]);
ylen=max([imag(Efd_gm);imag(Eron_gm)])*1.1;
xstd=std([real(Efd_gm);real(Eron_gm)]);
dxdx=max(3*xstd,ylen);
xlim([xm-3*xstd,xm+3*xstd])
ylim([-dxdx,dxdx])