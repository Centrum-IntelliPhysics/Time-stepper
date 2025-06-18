clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
%load('RandONet_single_AC_022.mat')
pod_modes=1;
if pod_modes==1
    load('POD_RandONet_parametric_FHN_0005_0995.mat')
    load('PODmodes.mat')
else
    load('RandONet_parametric_FHN_0005_0995.mat')
end
variation_on=0; %<---------------------
RandONet.C=gather(RandONet.C);
simmetry=0;
%load('')
%
x0=0; xL=20;
t0=0; 
Nx=41;
Nt=4501; %41;%31;
dtRON=0.1; %0.05;
tf=(Nt-1)*dtRON+t0; %tf=2; 40*0.05=2
xspan=linspace(x0,xL,Nx)'; %output grid
dx=xspan(2)-xspan(1);
tspan=linspace(t0,tf,Nt); %output times grid (define dt)
[XX,TT]=meshgrid(xspan,tspan);

%FHN parameters
% Setting parameters
Du = 1; %coefficient of diffusion activator
Dv = 4; %coefficient of diffusion inibitor
a0 = -0.03; %parameter of the problem
a1 = 2;
eps1=0.008;  %<------------------
dtLBM=0.001; %time step
%

%FHN finite difference
w_u_ic=0.9; a_u_ic=0.8; c_u_ic=11; b_u_ic=0.1;
u0=w_u_ic*tanh(a_u_ic*(xspan-c_u_ic))+b_u_ic;
w_v_ic=0.15; a_v_ic=0.4; c_v_ic=5; b_v_ic=0.02;
v0=w_v_ic*tanh(a_v_ic*(xspan-c_v_ic))+b_v_ic;
uv0=[u0;v0];

heal_time=5000;
nfdron=round(dtRON/dtLBM)+1;
%
fun_FD_FHN=@(uv0) FD_FHN(uv0,Du,Dv,a1,a0,eps1,dtLBM,Nx,dx);
tmsp_FD_FHN=@(uv0) time_stepper_end(fun_FD_FHN,uv0,nfdron);
eqs_FD_FHN=@(uv0) uv0-tmsp_FD_FHN(uv0);
uv1=time_stepper_end(fun_FD_FHN,uv0,heal_time);
%
true=time_stepper(tmsp_FD_FHN,uv1,Nt);

figure(1)
surf(XX,TT,true(1:Nx,:)','EdgeColor','none','LineStyle','none','FaceColor','interp')
clim([-1,1])
colorbar()
view(2)
folder='figures2pod/';
equal_name='fig_FHN_single_par_';
type=1;
%typename={'_rand'};
%
name='ref_space_time_traj_u1';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')


figure(2)
surf(XX,TT,true(Nx+1:end,:)','EdgeColor','none','LineStyle','none','FaceColor','interp')
clim([-1,1])
colorbar()
view(2)
name='ref_space_time_traj_v2';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%
%uv0=true(1,:);
if variation_on==1
    fun_RON=@(x) x+EVAL_flags_RandONet(RandONet,x,eps1,xspan,parametric);
else
    fun_RON=@(x) EVAL_flags_RandONet(RandONet,Vkov_proj*(x-bias_proj),eps1,xspan,parametric);
end
eqs_RON=@(uv0) uv0-fun_RON(uv0);
uvRON=time_stepper(fun_RON,uv1,Nt);
%uu=uu;
%toc
%
figure(3)
II=[1,20,50,100:100:Nt,Nt];
for i=II
plot(xspan,true(1:Nx,i),'b')
hold on
plot(xspan,uvRON(1:Nx,i),'--r')
end
ylim([-1.1,1.1])
pause(0.001)
name='RandONet_vs_ref';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%
figure(4)
surf(XX,TT,uvRON(1:Nx,:)','LineStyle','none','FaceColor','interp')
view(2)
colorbar()
clim([-1,1])
name='RandONet_space_time_traj_u1';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%
figure(5)
surf(XX,TT,uvRON(Nx+1:end,:)','LineStyle','none','FaceColor','interp')
view(2)
colorbar()
clim([-1,1])
name='RandONet_space_time_traj_v2';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%
figure(6)
hold off
contourf(XX,TT,abs(true(1:Nx,:)-uvRON(1:Nx,:))'+1e-15,10.^(-16:1:16),'LineWidth',0.01,'ShowText','off','LineColor',[.3 .3 .3],'LineStyle','none')
set(gca,'ColorScale','log')
colorbar()
c = colorbar;
%colormap(jet)
ax = gca;
c_ax=clim(ax);
%clim(ax,[c_ax(1)*10,max(1,c_ax(2))]);
clim([10.^-3,1])
c.Ticks = 10.^(-16:1:16);
c.Label.String = 'abs error';
c.Label.Interpreter = 'latex';
c.Label.FontSize = 18;
%
ax.FontName = 'Times';
ax.FontSize = 20;
%ax.LabelFontSizeMultiplier = 15/12;
xlabel('$x$','FontSize',20,'Interpreter','latex');
ylabel('$t$','FontSize',20,'Interpreter','latex','Rotation',0);
%
name='RandONet_space_time_error_u1';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')

figure(7)
hold off
contourf(XX,TT,abs(true(Nx+1:2*Nx,:)-uvRON(Nx+1:2*Nx,:))'+1e-15,10.^(-16:1:16),'LineWidth',0.01,'ShowText','off','LineColor',[.3 .3 .3],'LineStyle','none')
set(gca,'ColorScale','log')
colorbar()
c = colorbar;
%colormap(jet)
ax = gca;
c_ax=clim(ax);
%clim(ax,[c_ax(1)*10,max(1,c_ax(2))]);
clim([10.^-3,1])
c.Ticks = 10.^(-16:1:16);
c.Label.String = 'abs error';
c.Label.Interpreter = 'latex';
c.Label.FontSize = 18;
%
ax.FontName = 'Times';
ax.FontSize = 20;
%ax.LabelFontSizeMultiplier = 15/12;
xlabel('$x$','FontSize',20,'Interpreter','latex');
ylabel('$t$','FontSize',20,'Interpreter','latex','Rotation',0);
name='RandONet_space_time_error_v2';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')


pause(0.001)
%%%%%%%%%%%%%%%%%
uv00RON=uvRON(:,end);

[uv00nFD,errfFD,iterFD,Jfd]=newton_method(@(x) eqs_FD_FHN(x),uv00RON,1e-12,1e-10,1e-7,100);
%
[uv00nRON,errf,iter,Jron]=newton_method(@(x) eqs_RON(x),uv00nFD,1e-6,1e-6,1e-6,100);
%

figure(8)
hold off
plot(xspan,uv00nFD(1:Nx,1))
hold on
plot(xspan,uv00nFD(Nx+1:2*Nx,1))
plot(xspan,uv00nRON(1:Nx,1),'--')
plot(xspan,uv00nRON(Nx+1:2*Nx,1),'--')
legend('Euler FD u','Euler FD v', 'POD-RandONet u','POD-RandONet v','Location','northwest')
set(gca,'FontSize',20)
xlabel('$x$','Interpreter','latex')
ylabel('$u(x), \, v(x)$','Interpreter','latex')
grid on
name='RandONet_steady_states';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')

%%%%%%%%
Jron=eye(2*Nx)-Jron;
[Vron,Eron]=eig(Jron);
Eron=diag(Eron);
%
Jfd=eye(2*Nx)-Jfd;
[Vfd,Efd]=eig(Jfd);
Efd=diag(Efd);

theta=linspace(0,2*pi,1001);
%
figure(9)
hold off
plot(cos(theta)+1j*sin(theta),'--k','HandleVisibility','off')
hold on
plot(Efd,'*b')
plot(Eron,'or')
legend('Euler FD','POD-RandONet')
set(gca,'FontSize',20)
xlabel('$Re(\mu$)','Interpreter','latex')
ylabel('$Im(\mu$)','Interpreter','latex')
grid on
xlim([-1.1,1.1])
ylim([-1.1,1.1])
name='RandONet_full_spectrum';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')


epfd=1e-6;
Jvron=@(x) (fun_RON(uv00nRON+epfd*x)-fun_RON(uv00nRON))/epfd;
Jvfd=@(x) (tmsp_FD_FHN(uv00nFD+epfd*x)-tmsp_FD_FHN(uv00nFD))/epfd;
[Vron_gm,Eron_gm]=eigs(Jvron,2*Nx,[],3,'largestabs','Tolerance',1e-6);
[Vfd_gm,Efd_gm]=eigs(Jvfd,2*Nx,[],3,'largestabs','Tolerance',1e-8);
Eron_gm=diag(Eron_gm);
Efd_gm=diag(Efd_gm);
figure(10)
hold off
plot(cos(theta),sin(theta),'--k','HandleVisibility','off')
hold on
plot(real(Efd_gm),imag(Efd_gm),'*b')
plot(real(Eron_gm),imag(Eron_gm),'or')
legend('Euler FD','POD-RandONet')
set(gca,'FontSize',20)
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
name='RandONet_3_leading_eigs';
filename=[folder,equal_name,name];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')

figure(11)
subplot(1,2,1)
hold off
plot(cos(theta)+1j*sin(theta),'--k','HandleVisibility','off')
hold on
plot(Efd,'*b')
plot(Eron,'or')
legend('Euler FD','POD-RandONet')
set(gca,'FontSize',20)
xlabel('$Re(\mu$)','Interpreter','latex')
ylabel('$Im(\mu$)','Interpreter','latex')
grid on
xlim([-1.1,1.1])
ylim([-1.1,1.1])
%
subplot(1,2,2)
hold off
plot(cos(theta),sin(theta),'--k','HandleVisibility','off')
hold on
plot(real(Efd_gm),imag(Efd_gm),'*b')
plot(real(Eron_gm),imag(Eron_gm),'or')
legend('Euler FD','POD-RandONet')
set(gca,'FontSize',20)
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