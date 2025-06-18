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
Ngaps=20; %10; <-------------
Nteeth=21;% 11; %<------------
Np_teeth=21; %11; <------------
Np_gaps=21; %11; <------------
Nx=Np_teeth*Nteeth+Ngaps*(Np_gaps-2); %-2 becuase gaps share points with teeth
%
Nt=41; %61 %21 %41;%31;  <----------------------
dt=0.0001; %0.05;
tf=(Nt-1)*dt+t0; %tf=2; 40*0.05=2
xspan=linspace(x0,xf,Nx); %output grid
dx=xspan(2)-xspan(1);
tspan=linspace(t0,tf,Nt); %output times grid (define dt)
[XX,TT]=meshgrid(xspan,tspan);
%
Npar=39; %39; %number of parameters <----------------------
lambd0=0; %minimum lambda considered 0
lambdf=3.8; %maximum lambda considered 3.8
lambd=linspace(lambd0,lambdf,Npar); %step in lambd is 0.1;
%
symmetry=0; %symmetry =1 no symmetry, =2 yes symmetric IC
%
%N_IC_rep=5; %5  %number of repeated initial conditions <--------------
N_IC_add=10; %20; %5 ;%5 what is this? number of additional initial conditions <----------

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
        nc=20*rand(1); %integer frequency of cosine
        ns=20*rand(1); %integer frequency of sine
        Ac=rand(1)/3; %amplitude of cosine
        As=rand(1)/3; %amplitude of sine
        l0=3*rand(1);
        gc=rand(1)*4/6+1/6;
        gs=20*rand(1);
        pswitch=(randi([1,2],1,1)-1)*2-1;
        u0_gauss=@(x) exp(-gs*(x-gc).^2)+1/2;
        u0_pert=@(x) (Ac*cos(nc*2*pi*x)+1).*(As*sin((ns*2*pi)*x)+1);
        u0_f=@(x) pswitch*l0*u0_pert(x).*(x-x.^2).*u0_gauss(x);
        %
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

%Nt,Nx,Npar=size(utrue)
%want to have [u+lambda] in branch and y in output
%U size [m grid,samples], here utrue(1:Nt-1,:,:)
%V size [m grid, samples], here utrue(2:Nt,:,:)
%Y size [n,1] or [1,n], here
healing_t=5; %jump time points (can be 0)
Ns=N_IC*(Nt-1-healing_t)*(Nx+1-Np_teeth); %number of samples
%Ng=Nx*Npar; %
d=1;
U_branch=zeros(Np_teeth+3,Ns); %Nx+parameter!+x center patch
param_branch=zeros(1,Ns);
%X_trunk=zeros(Nx,d);
Y_trunk=linspace(0,(Np_teeth-1)*dx,Np_teeth)'; %xspan';
V_out=zeros(Np_teeth,Ns);
s=0;
for i=1:N_IC
    for jt=1:Nt-1-healing_t
        for kte=1:Nteeth
            kx=1+(kte-1)*Np_teeth+(kte-1)*(Np_gaps-2);
            kxp=1+(kte-2)*Np_teeth+(kte-2)*(Np_gaps-2);
            kxn=1+(kte)*Np_teeth+(kte)*(Np_gaps-2);
        %for kx=1:Nx+1-Np_teeth
            s=s+1;%kx+(jt-1)*(Nx-2-Np_teeth)+(Nx-2-Np_teeth)*(Nt-1-healing_t)*(i-1);
            xind=kx:kx+Np_teeth-1;
            xindp=kxp:kxp+Np_teeth-1;
            xindn=kxn:kxn+Np_teeth-1;
            if kte>1
                %U_branch(1,s)=(true(jt+healing_t,xind(1)+1,i)-true(jt+healing_t,xind(1)-1,i))/(2*dx);
                %U_branch(1,s)=true(jt+healing_t,xind(1)-1,i);
                U_branch(1,s)=mean( true(jt+healing_t,xindp,i) );
            else
                %−3/2 at x0	2 at x1	−1/2 at x2
                U_branch(1,s)=0;
                %U_branch(1,s)=(-3/2*true(jt+healing_t,xind(1)+2,i)...
                %    +2*true(jt+healing_t,xind(1)+1,i)...
                %    -3/2*true(jt+healing_t,xind(1),i))/(dx);
            end
            U_branch(2:Np_teeth+1,s)=true(jt+healing_t,xind,i)';
            if kte<Nteeth
                %U_branch(Np_teeth+2,s)=(true(jt+healing_t,xind(end)+1,i)-true(jt+healing_t,xind(end)-1,i))/(2*dx);
                %U_branch(Np_teeth+2,s)=true(jt+healing_t,xind(end)+1,i);
                U_branch(Np_teeth+2,s)=mean( true(jt+healing_t,xindn,i) );
            else
                %1/2 at x-2	−2 atx-1	3/2 at x0
                %U_branch(Np_teeth+2,s)=(3/2*true(jt+healing_t,xind(end),i)...
                %    -2*true(jt+healing_t,xind(end)-1,i)...
                %    +1/2*true(jt+healing_t,xind(end)-2,i))/(dx);
                U_branch(Np_teeth+2,2)=0;
            end
            %U_branch(Np_teeth+3,s)=save_par(i);
            U_branch(Np_teeth+3,s)=xspan(xind(round(Np_teeth/2)));
            param_branch(1,s)=save_par(i);
            V_out(:,s)=true(jt+1+healing_t,xind,i)';
        end
    end
end
%
Itr=randperm(Ns,floor(Ns*80/100));
param_train_branch=param_branch(:,Itr);
Utrain_branch=U_branch(:,Itr);
Ytrain_trunk=Y_trunk;
Vtrain_out=V_out(:,Itr);
U_branch(:,Itr)=[];
param_branch(:,Itr)=[];
V_out(:,Itr)=[];
%
Nsrest=size(U_branch,2);
Ival=randperm(Nsrest,floor(Nsrest*50/100));
%
Uval_branch=U_branch(:,Ival);
param_val_branch=param_branch(:,Ival);
Yval_trunk=Y_trunk;
Vval_out=V_out(:,Ival);
U_branch(:,Ival)=[];
param_branch(:,Ival)=[];
V_out(:,Ival)=[];
%
Utest_branch=U_branch;
param_test_branch=param_branch;
Ytest_trunk=Y_trunk;
Vtest_out=V_out;
%

save('data_00_38_Bratu_patches_chebfun',...
    'Utrain_branch','param_train_branch','Ytrain_trunk','Vtrain_out',...
    'Utest_branch','param_test_branch','Ytest_trunk','Vtest_out',...
    'Vval_out','Yval_trunk','Uval_branch','param_val_branch',...
    "Np_teeth","Np_gaps","Ngaps","Nteeth","Nx","dx","dt");