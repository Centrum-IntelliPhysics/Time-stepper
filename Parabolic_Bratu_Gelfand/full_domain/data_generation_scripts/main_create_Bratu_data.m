%create AC data
clear
close all
clc
set(0,'DefaultLineLineWidth',2)
%
plot_on=1; %<-------------------------
flag_aligned=1;
%
x0=0; xf=1;
t0=0; 
Nx=51;
Nt=31; %21 %41;%31;  <----------------------
dt=0.001; %0.05;
tf=(Nt-1)*dt+t0; %tf=2; 40*0.05=2
xspan=linspace(x0,xf,Nx); %output grid
tspan=linspace(t0,tf,Nt); %output times grid (define dt)
[XX,TT]=meshgrid(xspan,tspan);
%
Npar=39; %number of parameters <----------------------
lambd0=0; %minimum lambda considered 0
lambdf=3.8; %maximum lambda considered 3.8
lambd=linspace(lambd0,lambdf,Npar); %step in lambd is 0.1;
%
symmetry=1; %symmetry =1 no symmetry, =2 yes symmetric IC
%
%N_IC_rep=5; %5  %number of repeated initial conditions <--------------
N_IC_add=40; %5 ;%5 what is this? number of additional initial conditions <----------

%chebfun parameters
xcheb=chebfun('x',[x0,xf]);
bc.left=@(u) u;
bc.right=@(u) u;
optstrue=pdeset('RelTol',1e-6,'AbsTol',1e-8,'N',512);


save_par=[];
kIC=0;
for ipar=1:Npar
    %parameter of bifurcation
    lambda=lambd(ipar)
    %other solutions random
    for ij=1:N_IC_add
        kIC=kIC+1;
        nc=10*rand(1); %integer frequency of cosine
        ns=10*rand(1); %integer frequency of sine
        Ac=rand(1)/5; %amplitude of cosine
        As=rand(1)/5; %amplitude of sine
        l0=3*rand(1);
        gc=rand(1)/2+1/4;
        gs=10*rand(1);
        u0_gauss=@(x) exp(-gs*(x-gc).^2)+1/2;
        u0_pert=@(x) (Ac*cos(nc*2*pi*x)+1).*(As*sin((ns*2*pi)*x)+1);
        u0_f=@(x) l0*u0_pert(x).*(x-x.^2).*u0_gauss(x);
        u0cheb=u0_f(xcheb);
        fcheb=@(t,x,u) diff(u,2)+lambda*exp(u);
        %time space param IC
        uuchebtrue=pde15s(fcheb,tspan,u0cheb,bc,optstrue);
        if Nt==size(uuchebtrue,2)
            true(:,:,kIC)=uuchebtrue(xspan')';
            save_par(kIC)=lambda;
            %
            if plot_on==1
            figure(5)
            hold off
            plot(xspan,u0_f(xspan),'--')
            hold on
            %hold off
            plot(xspan,true(2,:,kIC))
            plot(xspan,true(5,:,kIC))
            plot(xspan,true(end,:,kIC))
            %ylim([-0.1,6])
            pause(0.001)
            end
        else
            kIC=kIC-1;
        end
    end
end

N_IC=size(true,3);

%input branch function v in grid (100)
%input trunk y and lambda
%output v(y)
if flag_aligned==0
healing_t=2;
Ns=Nx*N_IC*(Nt-1-healing_t);
Ng=Nx*Npar;

U_branch=zeros(Ns,Nx);
Y_trunk=zeros(Ns,2);
V_out=zeros(Ns,1);

for ip=1:N_IC
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
    %want to have [u+lambda] in branch and y in output
    %U size [m grid,samples], here utrue(1:Nt-1,:,:)
    %V size [m grid, samples], here utrue(2:Nt,:,:)
    %Y size [n,1] or [1,n], here
    healing_t=2;
    Ns=N_IC*(Nt-1-healing_t); %number of samples
    Ng=Nx*Npar; %
    d=1;
    U_branch=zeros(Nx+1,Ns); %Nx+parameter!
    %X_trunk=zeros(Nx,d);
    Y_trunk=xspan';
    V_out=zeros(Nx,Ns);
    for i=1:N_IC %parameter (not exactly but saved like this, already copies)
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

save('data_00_38_Bratu_chebfun_aligned_healing2','Utrain_branch','Ytrain_trunk','Vtrain_out',...
    'Utest_branch','Ytest_trunk','Vtest_out','Vval_out','Yval_trunk','Uval_branch');