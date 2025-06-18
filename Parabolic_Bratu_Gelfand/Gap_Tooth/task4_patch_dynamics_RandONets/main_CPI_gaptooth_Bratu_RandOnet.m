clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
load('RandONet_single_Bratu_patches_10.mat')
load('data_00_38_Bratu_patches_chebfun.mat','dt','Ngaps',...
    'dx','Np_gaps','Nteeth','Np_teeth','Nx')
%load('RandONet_parametric_Bratu_022_070.mat')
RandONet.C=gather(RandONet.C);
simmetry=0;
%load('')
%
x0=0; xf=1;
t0=0; 
%Nx=51;
Nt=6;
nt=50;
%dt=0.0001; %0.005;
DT=0.015;
NNt=1+Nt*(nt+round(DT/dt))+nt;
tf=Nt*(nt*dt+DT)+nt*dt+t0; %tf=2; 40*0.05=2
%tf=(Nt-1)*dt+t0; %tf=2; 40*0.05=2
xspan=linspace(x0,xf,Nx)'; %output grid
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

l0=0.05*0.1; %rand(1);
u0_f=@(x) 4*l0*x.*(1-x)*(0.3*sin(1*pi*xcheb)+1);
u0cheb=u0_f(xcheb);
tic;
uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
true=uuchebtrue(xspan);
toc
truemax=max(abs(true),[],2);
truemax_x=max(abs(diff(true,1,2)),[],2)/dx;
truemax_2x=max(abs(diff(true,2,2)),[],2)/dx.^2;

%RandONet patches
DXteeth=dx*(Np_teeth-1);
DXgaps=dx*(Np_gaps-1);
xpatches=zeros(Np_teeth*Nteeth,1);
xbordpatches=zeros(2*Nteeth,1);
xindbord=zeros(2*Nteeth,1);
xindpatch=zeros(Nteeth*Np_teeth,1);
%
ypatch=linspace(0,DXteeth,Np_teeth)';
for i=1:Nteeth
    ind=(1:Np_teeth)+(i-1)*Np_teeth;
    xbordpatches((1:2)+2*(i-1))=[0-dx;DXteeth+dx]+DXteeth*(i-1)+DXgaps*(i-1);
    xpatches(ind,1)=ypatch+DXteeth*(i-1)+DXgaps*(i-1);
    xindbord((1:2)+2*(i-1),1)=[1;Np_teeth]+Np_teeth*(i-1)+(Np_gaps-2)*(i-1);
    xindpatch(ind,1)=(1:Np_teeth)'+Np_teeth*(i-1)+(Np_gaps-2)*(i-1);
end
%
[XXpatch,TTpatch]=meshgrid(xpatches,tspan);


u0=true(:,1);%u0cheb(xspan');
u0gap=spline(xspan,u0,xpatches);
fun_RONgap=@(x) EVAL_RandONet_patch(RandONet,x,lambda,...
    ypatch,xpatches,xbordpatches,xindbord,xspan,...
    simmetry,parametric,flag_single,...
    DXteeth,dx,Nx,Nteeth,Ngaps,Np_teeth,Np_gaps);
%uu=time_stepper(fun_RON,u0ron,Nt);
[ttCPI,uuRONgap_CPI]=CPI(fun_RONgap,u0gap,nt,dt,Nt,DT);
uuRONgap_CPI=uuRONgap_CPI;
[XXcpi,TTcpi]=meshgrid(xspan,ttCPI);
[XXpatchCPI,TTpatchCPI]=meshgrid(xpatches,ttCPI);
toc

[uuRONgap]=time_stepper(fun_RONgap,u0gap,NNt);
%
uchebtrue_tt=pde15s(fcheb,ttCPI,u0cheb,bc,optstrue);
true_tt=uchebtrue_tt(xspan);

%

figure(1)
plot(xspan,true(:,1),'k-')
hold on
plot(xpatches,uuRONgap(:,1),'or')
hold on
plot(xpatches,uuRONgap_CPI(:,1),'.g')
%II=[2,5,10:10:Nt,Nt];
%II=[1:NNt];
for i=1:nt:NNt
    plot(xspan,true(:,i),'k','HandleVisibility','off')
    plot(xpatches,uuRONgap(:,i),'or','HandleVisibility','off')
end
for i=1:1:length(ttCPI)
    plot(xpatches,uuRONgap_CPI(:,i),'.g','HandleVisibility','off')
    pause(0.001)
end
legend('reference','Gap-Tooth RandONet','Patches RandONet','Location','northwest')
%ylim([-1.1,1.1])
xlabel('$x$','Interpreter','latex')
ylabel('$u(x)$','Interpreter','latex')
set(gca,'FontSize',18)
grid on
pause(0.001)
%
folder='figures/';
equal_name='fig_Bratu_PatchRON_gapCPI_single_par_';
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
for j=1:Nt
J=(1:nt+1)+(nt+1)*(j-1);
jend=(nt+1)*(j-1)+nt+1;
j0p=(nt+1)*(j)+1;
for i=1:Nteeth
    I=(1:Np_teeth)+Np_teeth*(i-1);
    surf(XXpatchCPI(J,I),TTpatchCPI(J,I),uuRONgap_CPI(I,J)','LineStyle','none','FaceColor','interp')
    hold on
    imid=I(11);
    v12=uuRONgap_CPI(imid,jend);
    if j<Nt
        quiver3(xpatches(imid),ttCPI(jend),uuRONgap_CPI(imid,jend),...
                0,DT,uuRONgap_CPI(imid,j0p)-uuRONgap_CPI(imid,jend),...
            'LineWidth',2, 'MaxHeadSize', 5,...
            'Color',[0.5,0.5,0.5],'LineStyle','-','AutoScaleFactor',1)
    end
    pause(0.001)
end
end
%view(2)
colorbar()
cc1=clim;%([0,0.2])
set(gca,'FontSize',20)
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
surf(XX,TT,true','LineStyle','none','FaceColor','interp')
view(2)
colorbar()
clim(cc1)
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
for j=1:Nt
    J=(1:nt+1)+(nt+1)*(j-1);
for i=1:Nteeth
    I=(1:Np_teeth)+Np_teeth*(i-1);
    contourf(XXpatchCPI(J,I),TTpatchCPI(J,I),abs(true_tt(xindpatch(I),J)-uuRONgap_CPI(I,J))'+1e-15,10.^(-16:1:16),'LineWidth',0.01,'ShowText','off','LineColor',[.3 .3 .3],'LineStyle','none')
    hold on
end
end
set(gca,'ColorScale','log','FontSize',18)
colorbar()
c = colorbar;
colormap(jet)
ax = gca;
c_ax=clim(ax);
%clim(ax,[min(c_ax(1),1e-5),max(1,c_ax(2))]);
clim([1e-5,1])
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
grid on
%
name='RandONet_space_time_error';
filename=[folder,equal_name,name,typename{type}];
saveas(gcf,[filename, '.eps'], 'epsc')
saveas(gcf,[filename, '.fig'])
saveas(gcf,[filename, '.jpg'], 'jpg')
saveas(gcf,[filename, '.pdf'], 'pdf')
%

return
%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%
%return
u00RON=uu(end,:)';
tmsp_RON=@(x) time_stepper(@(x) fun_RONgap(x),x,2,1);
eqs_RON=@(x) x-tmsp_RON(x);

%FD
lam0=1.0;
NxFD=length(xpatches); %grid for Finite difference
xspanFD=linspace(x0,xf,NxFD)';
u00FD=spline(xpatches,u00RON,xspanFD);
dxFD=xspanFD(2)-xspanFD(1);
dtFD=1e-6;%dxFD^2/4;
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
plot(xspanFD,u00nFD,'k')
hold on
plot(xpatches,u00nRON,'ro','MarkerSize',3)
legend('Euler FD','Gap-Tooth RandONet','Location','northwest')
set(gca,'FontSize',20)
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
Nxpatches=length(xpatches);
Jron=eye(Nxpatches)-Jron;
[Vron,Eron]=eig(Jron);
Eron=diag(Eron);
%
Jfd=eye(NxFD)-Jfd;
[Vfd,Efd]=eig(Jfd);
Efd=diag(Efd);
Efd=complex(Efd);
%
theta=linspace(0,2*pi,1001);
%
figure(8)
hold off
plot(cos(theta)+1j*sin(theta),'--k','HandleVisibility','off')
hold on
plot(Efd,'*b')
plot(Eron,'or','MarkerSize',4)
legend('Euler FD','Gap-Tooth RandONet')
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
[Vron_gm,Eron_gm]=eigs(Jvron,Nxpatches,[],3,'largestabs','Tolerance',1e-6);
[Vfd_gm,Efd_gm]=eigs(Jvfd,NxFD,[],3,'largestabs','Tolerance',1e-8);
Eron_gm=diag(Eron_gm);
Efd_gm=diag(Efd_gm);
figure(9)
hold off
plot(cos(theta),sin(theta),'--k','HandleVisibility','off')
hold on
plot(real(Efd_gm),imag(Efd_gm),'*b')
plot(real(Eron_gm),imag(Eron_gm),'or')
legend('Euler FD','Gap-Tooth RandONet')
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

fig10=figure(10);
fig10.Position(3)=1170;
%
subplot(1,2,1)
hold off
plot(cos(theta)+1j*sin(theta),'--k','HandleVisibility','off')
hold on
plot(Efd,'*b')
plot(Eron,'or','MarkerSize',4)
legend('Euler FD','Gap-Tooth RandONet')
set(gca,'FontSize',18)
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
plot(real(Efd_gm),imag(Efd_gm),'*b','MarkerSize',10)
plot(real(Eron_gm),imag(Eron_gm),'or','MarkerSize',10)
legend('Euler FD','Gap-Tooth RandONet')
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

% Create rectangle
annotation(fig10,'rectangle',...
    [0.444228454172367 0.523809523809524 0.0133638850889194 0.051428571428573],...
    'Color',[0.501960784313725 0.501960784313725 0.501960784313725],...
    'LineStyle','-.');