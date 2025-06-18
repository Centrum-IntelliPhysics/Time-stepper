%create AC data
clear
close all
clc
set(0,'DefaultLineLineWidth',2)
%
plot_on=1; %<-------------------------
flag_aligned=1;
%
x0=-1; xf=1;
t0=0; 
Nx=100;
Nt=41; %21 %41;%31;  <----------------------
dt=0.01; %0.05;
tf=(Nt-1)*dt+t0; %tf=2; 40*0.05=2
xspan=linspace(x0,xf,Nx); %output grid
tspan=linspace(t0,tf,Nt); %output times grid (define dt)
[XX,TT]=meshgrid(xspan,tspan);
%
Npar=49; %49 %25;%22; %number of values for parameter <----------------------
nmax=2; %maximum number of bifurcation levels
nn=1:nmax; %from 1 to max number of bifurcations
%
epss_bif_sin=1./(pi/2+(nn-1)*pi); %exact bifurccation point of type 1(sin)
epss_bif_cos=1./(nn*pi); %exact bifurcation point of type 2 (cos)
%
epss0=ceil(epss_bif_sin(2)*100)/100; %minimum epsilon considered 0.22
epssf=ceil(epss_bif_sin(1)*10)/10; %maximum epsilon considered 0.7
epss=linspace(epss0,epssf,Npar); %step in epss is 0.02; 0.02*(25-1)=0.48, 0.22+0.48=0.7
%
%basically it will use both u(x) and -u(x) as input (simmetry for
%pitchfork)
%
%note: NOT u(x) and u(-x) as cos is symmetric, sin is not symmetric
flag_symmetry=2; %flag_symmetry =1 no symmetry data, =2 yes symmetric data IC
%
N_IC_rep=5; %5  %number of repeated initial conditions <--------------
N_IC_add=5; %5  %number of additional initial conditions <----------

%Chebychev pseudo-spectral solver (chebfun package, need to install)
%chebfun parameters
xcheb=chebfun('x',[x0,xf]);
bc.left=@(u) diff(u,1);
bc.right=@(u) diff(u,1);
optstrue=pdeset('RelTol',1e-6,'AbsTol',1e-8,'N',512);

%true=zeros(Nt,Nx,Npar,N_IC*symmetry);
%offset=0.01;
save_par=[];
kIC=0;
for ipar=1:Npar
    %parameter of bifurcation
    epsilon=epss(ipar)
    %true sol
    %sine solutions
    nk_sin=sum(epss_bif_sin>epsilon); %number of sine bif solutions in the range
    for nj=1:nk_sin+1
        for nit=1:N_IC_rep
        kIC=kIC+1;
        sb=1.1*sqrt(abs(epsilon-epss_bif_sin(nj))); %approximated sb far from bifurcation point
        sb=sb*(1+(2*rand(1)-1)/10); %perturbed randomly
        %[kIC,nj,sb]
        nc=randi(5); %integer frequency of cosine
        ns=randi(5); %integer frequency of sine
        Ac=0.1*rand(1); %amplitude of cosine
        As=0.1*rand(1); %amplitude of sine
        u0_f_noise=@(x) (Ac*cos(nc*pi*x)+As*sin((ns*pi+pi/2)*x));
        u0_f=@(x) sb*sin(((nj-1)*pi+pi/2)*x)+u0_f_noise(x); %typical solution+noise
        u0cheb=u0_f(xcheb);
        %CHEBFUN pde15s solver (adaptive BDF in time integration)
        fcheb=@(t,x,u) epsilon*diff(u,2)+1/epsilon*(u-u.^3);
        uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
        %(time, space, param, IC)
        true(:,:,kIC)=uuchebtrue(xspan')';
        save_par(kIC)=epsilon;
        %
        if plot_on==1
        figure(5)
        hold off
        plot(xspan,u0_f(xspan))
        hold on
        plot(xspan,u0_f_noise(xspan),'--')
        pause(0.001)
        %
        figure(6)
        %hold off
        plot(xspan,true(end,:,kIC))
        hold on
        plot(xspan,true(2,:,kIC))
        plot(xspan,true(5,:,kIC))
        ylim([-1.1,1.1])
        pause(0.001)
        end
        end
    end
    %cosine solutions
    nk_cos=sum(epss_bif_cos>epsilon); %number of cosine bif solutions in the range
    for nj=1:nk_cos+1
        for nit=1:N_IC_rep
        kIC=kIC+1;
        sb=1.1*sqrt(abs(epsilon-epss_bif_cos(nj)));
        sb=sb*(1+(2*rand(1)-1)/10);
        %
        nc=randi(5); %integer frequency of cosine
        ns=randi(5); %integer frequency of sine
        Ac=0.1*rand(1); %amplitude of cosine
        As=0.1*rand(1); %amplitude of sine
        u0_f_noise=@(x) (Ac*cos(nc*pi*x)+As*sin((ns*pi+pi/2)*x));
        u0_f=@(x) sb*cos(nj*pi*x)+u0_f_noise(x); %typical solution + noise
        u0cheb=u0_f(xcheb);
        fcheb=@(t,x,u) epsilon*diff(u,2)+1/epsilon*(u-u.^3);
        %(time, space, param, IC)
        uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
        true(:,:,kIC)=uuchebtrue(xspan')';
        save_par(kIC)=epsilon;
        %
        if plot_on==1
        figure(5)
        %hold off
        plot(xspan,u0_f(xspan))
        pause(0.001)
        %
        figure(6)
        %hold off
        plot(xspan,true(end,:,kIC))
        hold on
        plot(xspan,true(2,:,kIC))
        plot(xspan,true(5,:,kIC))
        ylim([-1.1,1.1])
        pause(0.001)
        end
        %::::
        end
    end
    %zero unstable solution
    kIC=kIC+1;
    u0_f=@(x) 0*x;%offset+(Ac*cos(nc*pi*x)+As*sin((ns*pi+pi/2)*x));
    u0cheb=u0_f(xcheb);
    fcheb=@(t,x,u) epsilon*diff(u,2)+1/epsilon*(u-u.^3);
    %time space param IC
    uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
    true(:,:,kIC)=uuchebtrue(xspan')';
    save_par(kIC)=epsilon;
   
    for nic=1:N_IC_rep
    %one flat solution
    kIC=kIC+1;
    nc=randi(5); %integer frequency of cosine
    ns=randi(5); %integer frequency of sine
    Ac=0.1*rand(1); %amplitude of cosine
    As=0.1*rand(1); %amplitude of sine
    u0_f_noise=@(x) (Ac*cos(nc*pi*x)+As*sin((ns*pi+pi/2)*x));
    u0_f=@(x) 3*rand(1)-1.5+0*x+u0_f_noise(x);
    u0cheb=u0_f(xcheb);
    fcheb=@(t,x,u) epsilon*diff(u,2)+1/epsilon*(u-u.^3);
    %time space param IC
    uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
    true(:,:,kIC)=uuchebtrue(xspan')';
    save_par(kIC)=epsilon;
    end
    
    %other kind of solutions (random cos and sin)
    for ij=1:N_IC_add
        kIC=kIC+1;
        nc=randi(5); %integer frequency of cosine
        ns=randi(5); %integer frequency of sine
        Ac=rand(1); %amplitude of cosine
        As=rand(1); %amplitude of sine
        offset=(2*rand(1)-1)/10;
        u0_f=@(x) offset+(Ac*cos(nc*pi*x)+As*sin((ns*pi+pi/2)*x));
        u0cheb=u0_f(xcheb);
        fcheb=@(t,x,u) epsilon*diff(u,2)+1/epsilon*(u-u.^3);
        %time space param IC
        uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
        true(:,:,kIC)=uuchebtrue(xspan')';
        save_par(kIC)=epsilon;
        %
        if plot_on==1
        figure(5)
        %hold off
        plot(xspan,u0_f(xspan))
        pause(0.001)
        %
        figure(6)
        %hold off
        plot(xspan,true(end,:,kIC))
        hold on
        plot(xspan,true(2,:,kIC))
        plot(xspan,true(5,:,kIC))
        ylim([-1.1,1.1])
        pause(0.001)
        end
    end
    
    %second class other solutions random (and variegated sum of Gaussians)
    for ij=1:N_IC_add
        kIC=kIC+1;
        offset=(2*rand(1)-1)/10;
        cen=2*rand(1,10)-1;
        shape=6*rand(1,10);
        wo=3*rand(1,10)-1.5;
        frand=@(x) wo(1)*(exp(-shape(1).*(x-cen(1)).^2))+...
            wo(2)*(exp(-shape(2).*(x-cen(2)).^2))+...
            wo(3)*(exp(-shape(3).*(x-cen(3)).^2))+...
            wo(4)*(exp(-shape(4).*(x-cen(4)).^2))+...
            wo(5)*(exp(-shape(5).*(x-cen(5)).^2))+...
            wo(6)*(exp(-shape(6).*(x-cen(6)).^2))+...
            wo(7)*(exp(-shape(7).*(x-cen(7)).^2))+...
            wo(8)*(exp(-shape(8).*(x-cen(8)).^2))+...
            wo(9)*(exp(-shape(9).*(x-cen(9)).^2))+...
            wo(10)*(exp(-shape(10).*(x-cen(10)).^2));
        u0_f=@(x) offset+(1+cos(pi*x))/2.*frand(x);
        u0cheb=u0_f(xcheb);
        fcheb=@(t,x,u) epsilon*diff(u,2)+1/epsilon*(u-u.^3);
        %time space param IC
        uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
        true(:,:,kIC)=uuchebtrue(xspan')';
        save_par(kIC)=epsilon;
        %
        if plot_on==1
        figure(5)
        %hold off
        plot(xspan,u0_f(xspan))
        pause(0.001)
        %
        figure(6)
        %hold off
        plot(xspan,true(end,:,kIC))
        hold on
        plot(xspan,true(2,:,kIC))
        plot(xspan,true(5,:,kIC))
        ylim([-1.1,1.1])
        pause(0.001)
        end
    end
end

N_IC=size(true,3);

if flag_symmetry==2
    true(:,:,(1:N_IC)+N_IC)=-true(:,:,1:N_IC);
    save_par((1:N_IC)+N_IC)=save_par(1:N_IC);
end

%input branch function v in grid (100)
%input trunk y and lambda
%output v(y)
if flag_aligned==0
healing_t=3;
Ns=Nx*N_IC*2*(Nt-1-healing_t);
Ng=Nx*Npar;

U_branch=zeros(Ns,Nx);
Y_trunk=zeros(Ns,2);
V_out=zeros(Ns,1);

for ip=1:2*N_IC
    for jt=1:Nt-1
        for kx=1:Nx
            s=kx+(jt-1)*Nx+(ip-1)*Nx*(Nt-1);
            U_branch(s,:)=true(jt+healing_t,:,ip);
            Y_trunk(s,1)=xspan(kx);
            Y_trunk(s,2)=save_par(ip);
            V_out(s,1)=true(jt+healing_t+1,kx,ip);
        end
    end
end

Itr=randperm(Ns,floor(Ns*80/100));
Utrain_branch=U_branch(Itr,:);
Ytrain_trunk=Y_trunk(Itr,:);
Vtrain_out=V_out(Itr,:);
U_branch(Itr,:)=[];
Y_trunk(Itr,:)=[];
V_out(Itr,:)=[];

Utest_branch=U_branch;
Ytest_trunk=Y_trunk;
Vtest_out=V_out;
%
elseif flag_aligned==1
    %Nt,Nx,Npar=size(utrue)
    %want to have [u+epsilon] in branch and y in output
    %U size [m grid,samples], here utrue(1:Nt-1,:,:)
    %V size [m grid, samples], here utrue(2:Nt,:,:)
    %Y size [n,1] or [1,n], here
    healing_t=3;
    Ns=N_IC*2*(Nt-1-healing_t); %number of samples
    Ng=Nx*Npar; %
    d=1;
    U_branch=zeros(Nx+1,Ns); %Nx+parameter!
    %X_trunk=zeros(Nx,d);
    Y_trunk=xspan';
    V_out=zeros(Nx,Ns);
    for i=1:2*N_IC %parameter (not exactly but saved like this, already copies)
        for jt=1:Nt-1-healing_t
            s=jt+(Nt-1-healing_t)*(i-1);
            U_branch(:,s)=[true(jt+healing_t,:,i)';save_par(i)];
            V_out(:,s)=true(jt+1+healing_t,:,i)';
        end
    end
    %
    Itr=randperm(Ns,floor(Ns*80/100));
    Utrain_branch=U_branch(:,Itr);
    Ytrain_trunk=Y_trunk;
    Vtrain_out=V_out(:,Itr);
    U_branch(:,Itr)=[];
    V_out(:,Itr)=[];
    %
    Nsrest=size(U_branch,2);
    Ival=randperm(Nsrest,floor(Nsrest*50/100));
    %
    Uval_branch=U_branch(:,Ival);
    Yval_trunk=Y_trunk;
    Vval_out=V_out(:,Ival);
    U_branch(:,Ival)=[];
    V_out(:,Ival)=[];
    %
    Utest_branch=U_branch;
    Ytest_trunk=Y_trunk;
    Vtest_out=V_out;
    %
end

save('data_022_070_AC_chebfun_more2_new_aligned_healing_shorter3','Utrain_branch','Ytrain_trunk','Vtrain_out',...
    'Utest_branch','Ytest_trunk','Vtest_out','Vval_out','Yval_trunk','Uval_branch');