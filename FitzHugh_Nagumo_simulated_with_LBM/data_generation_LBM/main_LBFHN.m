clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
rng(5)
% Lattice D1Q3
N=40; %number of sub_intervals (space-steps). (N+1) points
np=N+1; %number of points
x0=0;
xL=20; %size of the domain (0,20)
x=linspace(x0,xL,np); %equispaced grid
dx=xL/N; %distance between two points
x=x';
Ytrunk=x;
plot_on=0;
%x0=0:xL/40:xL;
% Setting parameters
Du = 1; %coefficient of diffusion activator
Dv = 4; %coefficient of diffusion inibitor
a0 = -0.03; %parameter of the problem
a1 = 2;
dtLBM=0.01; %time step
%
w=[1/6 4/6 1/6]; %optimal weights (LBM)
cs_quad=w(1)+w(3);
rel=[2/(1+2/cs_quad*Du*dtLBM/dx^2) 2/(1+2/cs_quad*Dv*dtLBM/dx^2)];

numb_epsilons=40;
epsilons=Cheby(0.005,0.995,numb_epsilons,2); %points of the second tipe: Chebychev-Gauss-Lobatto

%parameter for change initial condition
numb_initial=50; %20
%randi([0,1],1)*2-1 random sign
%s_sign=(randi([0,1],numb_epsilons,numb_initial)*2-1);
%w_u_ic=(0.8+0.4*rand(numb_epsilons,numb_initial));
%
w_u_ic=2.6*rand(numb_epsilons,numb_initial)-1.3;
a_u_ic=0.5+0.5*rand(numb_epsilons,numb_initial);
c_u_ic=2+16*rand(numb_epsilons,numb_initial);
b_u_ic=0.2*(2*rand(numb_epsilons,numb_initial)-1);
sw_u_ic=randi(3,numb_epsilons,numb_initial);
cw_u_ic=randi(3,numb_epsilons,numb_initial);
As_u_ic=0.1*rand(numb_epsilons,numb_initial);
Ac_u_ic=0.1*rand(numb_epsilons,numb_initial);
%
w_v_ic=0.6*rand(numb_epsilons,numb_initial)-0.3;
a_v_ic=0.5+0.5*rand(numb_epsilons,numb_initial);
c_v_ic=2+16*rand(numb_epsilons,numb_initial);
b_v_ic=0.05*(2*rand(numb_epsilons,numb_initial)-1);
sw_v_ic=randi(4,numb_epsilons,numb_initial);
cw_v_ic=randi(4,numb_epsilons,numb_initial);
As_v_ic=0.05*rand(numb_epsilons,numb_initial);
Ac_v_ic=0.05*rand(numb_epsilons,numb_initial);
%
%
fun_pert_ic=@(x,sw,cw,As,Ac) As*sin(sw*pi*(x-x0)/xL+pi/2)+...
   Ac*cos(cw*pi*(x-x0)/xL);
%
fun_ic= @(x,w,a,c,b,sw,cw,As,Ac) w*tanh(a.*(x-c))+b+fun_pert_ic(x,sw,cw,As,Ac);
%
fun_ic_NeuC=@(x,w,a,c,b,sw,cw,As,Ac) fun_ic(x,w,a,c,b,sw,cw,As,Ac)...
    -(fun_ic(xL,w,a,c,b,sw,cw,As,Ac)-fun_ic(xL-0.0001,w,a,c,b,sw,cw,As,Ac))/0.0001.*(x-x0)/(xL-x0)...
    -(fun_ic(x0+0.0001,w,a,c,b,sw,cw,As,Ac)-fun_ic(x0,w,a,c,b,sw,cw,As,Ac))/0.0001.*(xL-x)/(xL-x0);
[EPS,tilde]=meshgrid(epsilons,zeros(1,numb_initial));
for i=1:numb_epsilons
    for j=1:numb_initial
        uu_ic=fun_ic_NeuC(x,w_u_ic(i,j),a_u_ic(i,j),c_u_ic(i,j),...
            b_u_ic(i,j),sw_u_ic(i,j),cw_u_ic(i,j),As_u_ic(i,j),...
            Ac_u_ic(i,j));
        MEANu(i,j)=mean(uu_ic);
        if i<5 && j<5
            figure(13)
            plot(x,uu_ic,'--k')
            hold on
            pause(0.001)
        end
    end
end
if plot_on==1
figure(2)
plot(EPS,MEANu','o')
end

t0=0;
tf=20; %final time
tf_long=300;
UV0_train_branch=[]; UV0_test_branch=[];
PARAM_train_branch=[];
PARAM_test_branch=[];
U1_train_out=[]; U1_test_out=[];
V1_train_out=[]; V1_test_out=[];
for hh=1:length(epsilons) 
for kk=1:numb_initial %n_par-1 training set + test set
t=0;
%
%yu=w_u_ic(hh,kk)*tanh(a_u_ic(hh,kk)*(x-c_u_ic(hh,kk)))+b_u_ic(hh,kk);
yu=fun_ic_NeuC(x,w_u_ic(hh,kk),a_u_ic(hh,kk),c_u_ic(hh,kk),...
            b_u_ic(hh,kk),sw_u_ic(hh,kk),cw_u_ic(hh,kk),As_u_ic(hh,kk),...
            Ac_u_ic(hh,kk));
fprintf('eps=%2.4f, mean_u=%2.4f\n',epsilons(hh),mean(yu))
%yv=-w2_ic(kk)*tanh(a2_ic(kk)*(x-c2_ic(kk)))+b2_ic(kk);
%yv=0.12*yu;
yv=fun_ic_NeuC(x,w_v_ic(hh,kk),a_v_ic(hh,kk),c_v_ic(hh,kk),...
            b_v_ic(hh,kk),sw_v_ic(hh,kk),cw_v_ic(hh,kk),As_v_ic(hh,kk),...
            Ac_v_ic(hh,kk));
eps1=epsilons(hh); %bifurcation parameter
it=1; %iteration
%
%LBM iteration steps
if kk<5
    tf_temp=tf_long;
else
    tf_temp=tf;
end
vecact=zeros(ceil(tf_temp/dtLBM),np); %aactivator
vecin=zeros(ceil(tf_temp/dtLBM),np); %inibitor
tvec=zeros(ceil(tf_temp/dtLBM),1);%time
if plot_on==1
figure(30)
if kk<numb_initial
    plot(x,yu,'linewidth',2)
    hold on
    leg30{kk}=sprintf('training %d',kk);
else
    plot(x,yu,'k','linewidth',2)
    leg30{kk}='test';
end
end

if plot_on==1
figure(31)
hold on
axis([x0 xL -1.5 1.5])
if kk<numb_initial
    plot(x,yv,'linewidth',2)
    hold on
    leg31{kk}=sprintf('training %d',kk);
else
    plot(x,yv,'k','linewidth',2)
    leg31{kk}='test';
end
end
f1=yu*w(1); %f distribution of activator
f0=yu*w(2);
f_1=yu*w(3);
dact=f0(1,1)+f_1(1,1)+f1(1,1); %density activator
vecact(1,:)=dact;
tvec(1,:)=t;
f1(:,2)=w(1)*yv; %f distribution of inibitor
f0(:,2)=w(2)*yv;
f_1(:,2)=w(3)*yv;
din=f0(:,2)+f_1(:,2)+f1(:,2); %density inibitor
vecin(1,:)=din;
while t<tf_temp-1e-7
    [f1,f_1,f0]=LatticeBM(f1,f_1,f0,rel,a1,a0,eps1,w,dtLBM);    
    t=t+dtLBM;
    it=it+1;        
    dact=f0(:,1)+f_1(:,1)+f1(:,1);
    din=f0(:,2)+f_1(:,2)+f1(:,2);
    vecact(it,:)=dact;
    vecin(it,:)=din;
    tvec(it)=t;
    if mod(it,800*ceil(0.01/dtLBM))==1
    if plot_on==1
    figure(10)
    subplot(1,2,1)
    plot(x,dact,'o-')
    xlim([x0 xL])
    ylim([-2 2])
    title(sprintf('time t=%3.2f',t))
    xlabel('x')
    ylabel('u')
    subplot(1,2,2)
    plot(x,din,'o-')
    xlim([x0 xL])
    ylim([-2 2])
    title(sprintf('time t=%3.4f',t))
    xlabel('x')
    ylabel('v')
    pause(0.000001)
    end
    end
end
%
if plot_on==1
figure(1)
hold off
subplot(1,2,1)
hold off
surf(tvec(1:10:end),x,vecact(1:10:end,1:N+1)','FaceColor','interp',...
    'EdgeColor','none','FaceLighting','phong')
Ix = xlabel('$t$');Iy = ylabel('$x$');Iz = zlabel('$u$');
cx=get(Ix,'children'); cy=get(Iy,'children'); cz=get(Iz,'children');
set(Ix,'interpreter','latex');set(Ix,'FontSize',16);
set(Iy,'interpreter','latex');set(Iy,'FontSize',16);
set(Iz,'interpreter','latex');set(Iz,'FontSize',16);
colorbar()
view(2)
axis([t0 tf_temp x0 xL])
hold on
end
if kk==numb_initial
tt=tvec(1:1000:end);
xx=x(1:N+1);
[TT,XX]=meshgrid(tt,xx);
if plot_on==1
plot3(TT,XX,ones(size(XX))*10,'r.','linewidth',2) %vert
plot3(TT,XX,ones(size(XX))*10,'r.','linewidth',2) %orizz
view(2)
end
end
if plot_on==1
subplot(1,2,2)
hold off
surf(tvec(1:10:end),x,vecin(1:10:end,1:N+1)','FaceColor','interp',...
    'EdgeColor','none','FaceLighting','phong')
Ix = xlabel('$t$');Iy = ylabel('$x$');Iz = zlabel('$v$');
cx=get(Ix,'children'); cy=get(Iy,'children'); cz=get(Iz,'children');
set(Ix,'interpreter','latex');set(Ix,'FontSize',16);
set(Iy,'interpreter','latex');set(Iy,'FontSize',16);
set(Iz,'interpreter','latex');set(Iz,'FontSize',16);
colorbar()
view(2)
axis([t0 tf_temp x0 xL])
end

%Sampling from the solution
%esclude first 2 seconds
val_step=1/(10*dtLBM);%/dtLBM; %%dtLBM));
val_heal=500;
val_timestep=1/(10*dtLBM);
dtRON=val_timestep*dtLBM;
%
u0=vecact(val_heal+1:val_step:end-val_timestep,:)';
v0=vecin(val_heal+1:val_step:end-val_timestep,:)';
u1=vecact(val_heal+1+val_timestep:val_step:end,:)';
v1=vecin(val_heal+1+val_timestep:val_step:end,:)';
nt=size(u0,2);
if nt>size(u1,2)
    u0=u0(:,1:end-1);
    v0=v0(:,1:end-1);
    nt=size(u1,2);
end
UV0=[u0;v0];
param=eps1*ones(1,nt);
if kk<numb_initial
    UV0_train_branch=[UV0_train_branch,UV0];
    U1_train_out=[U1_train_out,u1];
    V1_train_out=[V1_train_out,v1];
    PARAM_train_branch=[PARAM_train_branch,param];
else
    UV0_test_branch=[UV0_test_branch,UV0];
    U1_test_out=[U1_test_out,u1];
    V1_test_out=[V1_test_out,v1];
    PARAM_test_branch=[PARAM_test_branch,param];
end
%
%disp(hh,kk)
end
end
%
if plot_on==1
figure(30)
legend(leg30)
xlabel('x')
ylabel('u')
h1=gca;
set(h1,'FontSize',16);
figure(31)
legend(leg31)
xlabel('x')
ylabel('v')
h1=gca;
set(h1,'FontSize',16);
end
%
save('data_LBM_FHN_dt01.mat','UV0_test_branch','UV0_train_branch',...
    'U1_train_out','U1_test_out','V1_test_out','V1_train_out',...
    "PARAM_test_branch",'PARAM_train_branch','Ytrunk','dtRON','dtLBM')