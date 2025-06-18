% train_RandONet trains a Random Projection-based Operator Network (RandONet) model.
%
% Syntax: net = train_RandONet(ff, yy, Nt, Nb, kmodel)
%
% Inputs:
%   - ff      : Input matrix (functions) for the branch network.
%   - yy      : Input vector (spatial locations) for the trunk network.
%   - G       : Input matrix (transformed functions G[f](y)
%   - Nt      : Number of neurons in the trunk network (default: 200).
%   - Nb      : Number of neurons in the branch network (default: 1000).
%   - kmodel  : Model type (1 for JL, 2 for RFFN; default: 2).
%
% Output:
%   - net     : Trained RandONet model containing fields for the trunk and
%               branch networks, including weights and biases.
%
%   - net : Structure containing the parameters of the RandONet model.
%           Fields include:
%             - tr_fT : Trunk network activation function (nonlinear transformation).
%             - tr_fB : Branch network activation function (nonlinear transformation).
%             - alphat, betat : Parameters for input transformation in the trunk network.
%             - alphab, betab : Parameters for input transformation in the branch network.
%             - C : Weight matrix for the inner product.
%
% The function initializes network parameters, trains using COD-based pseudo-inverse 
% of the trunk and branch layers, and stores the results in the output net.
%
% DISCLAIMER: This software is provided "as is" without warranty of any kind.
% This includes, but is not limited to, warranties of merchantability,
% fitness for a particular purpose, and non-infringement.
% The authors and copyright holders are not liable for any claims, damages,
% or other liabilities arising from the use of this software.
%
%Copyright (c) 2024 Gianluca Fabiani
%
%Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
% You may not use this material for commercial purposes.
% If you remix, transform, or build upon this material,
% you must distribute your contributions under the same license as the original.

function net=train_RandONet_parametric(ff,param,yy,G,Nt,Nb,kmodel,gpu_flag,n_out,identity)
%check arguments %basic selection
Ny=size(yy,1); %number of points of the evaluation grid
Nyg=size(G,1);
if nargin<5
    Nt=200;
end
if nargin<6
    Nb=1000;
end
if nargin<7
    kmodel=2; %RFFN
end
if nargin<8
    gpu_flag=1;
end
if nargin<9
    n_out=Nyg/Ny;
end
if nargin<10
    identity=0;
end
if isempty(Nt)
    Nt=200;
end
if isempty(Nb)
    Nb=1000;
end
if isempty(kmodel)
    kmodel=2;
end
if isempty(gpu_flag)
    gpu_flag=1;
end
%
%check
if n_out~=Nyg/Ny
    disp('Error in number of outputs')
    return
end
%
%main part of training
tr_fT=@(x) tanh(x); %activation function of the trunk network (only this one tested)
tr_fT_name='tanh';
if kmodel==1
    tr_fB=@(x) x; flag_branch=1; %%%%%%%JL based RandONet model
    tr_fB_name='lin';
elseif kmodel==2
    tr_fB=@(x) cos(x); flag_branch=0;  %%%%%%RFFN based RandONet model
    tr_fB_name='cos';
end
param0=min(param);
paramf=max(param);
dparam=paramf-param0;
if dparam<1e-6
    dparam=1e-6;
end
ff=[ff;param];
param_rescaled=(param-param0)/dparam;
net.param0=param0;
net.dparam=dparam;
Nx=size(ff,1); %number of discretization points of functions
Ns=size(ff,2); %number of samples
%
x0=min(yy); xf=max(yy); %spatial interval
DX=xf-x0; %size of the interval
alphat=2*(4+Nt*9/100)*(2*rand(1,Nt)-1)/(DX); %random internal weights of the trunk
cent=linspace(x0,xf,Nt); %
betat=-alphat.*cent; % random biases of the trunk
Tr=tr_fT(yy*alphat+betat); %trunk hidden layer
%branch net
U0=min(min(ff,[],2));
Uf=max(max(ff,[],2));
DU=Uf-U0; %normalization of the function space
%U0=U0-DU/10;
%Uf=Uf+DU/10;
if flag_branch==1 %JL
    alphab0=sqrt(1/Nb)*randn(Nb,Nx)/DU; %JL random internal weights in the branch
    alphab1=sqrt(1/Nb)*randn(Nb,Nx)/DU; %JL random internal weights in the branch
    betab=zeros(Nb,1);
end
if flag_branch==0
    alphab0=2*sqrt(Nx)*Nb^(1/8)*sqrt(2/Nb)*randn(Nb,Nx)/DU.*(rand(Nb,Nx)); %RFFN random internal weights in the branch
    alphab1=2*sqrt(Nx)*Nb^(1/8)*sqrt(2/Nb)*randn(Nb,Nx)/DU.*(rand(Nb,Nx)); %RFFN random internal weights in the branch
    betab=rand(Nb,1)*2*pi;
end
%         if flag_branch==2 (not tested)
%             alphab=1/Nx*(4+Nb.^(1/Nx)*9/100)*(2*rand(Nb,Nx)-1)./DU;
%             cenb=U0+DU.*rand(Nb,Nx);
%             betab=-sum(alphab.*cenb,2);
%         end
flag_norm=0;
if flag_norm==1
    norm_e=@(e) sqrt(e.^2+(1-e).^2);
else
    norm_e=@(e) 1;
end
Br_train=tr_fB(alphab0*ff.*((1-param_rescaled)./norm_e(param_rescaled))+...
    alphab1*ff.*((param_rescaled)./norm_e(param_rescaled))+betab); %branch hidden layer
if identity==1
    Br_train=[Br_train; ff];
end
%Br_train=Br_train;
%Br_train=[Br_train;param_rescaled];
%
if (Nb^2*Ns+Ns^2+Nb)<2*300^3 || gpu_flag==0
    tic;
    [Qr,T11,Vp]=pinvCOD(Tr); %COD based pseudo inverse of the trunk hidden layer
    [Qr2,T112,Vp2]=pinvCOD(Br_train); %COD based pseudo inverse of the branch hidden layer
    %
    flag_check=1;
    toc
else
    flag_check=0;
    tic;
    %Tr=gpuArray(Tr);
    [Qr,T11,Vp]=pinvCOD(Tr);
    Br=gpuArray(Br_train);
    tol=min(max(eps(norm(Br_train)*max(size(Br_train)))*1e1,1e-10),1e-4);
    %tol=1e-8;
    invBr=pinv(Br,tol);
    %C=pinv(Tr,1e-8)*G*pinv(Br,1e-8);
    % [Qr,T11,Vp]=pinvCOD(Tr); %COD based pseudo inverse of the trunk hidden layer
    % [Qr2,T112,Vp2]=pinvCOD(Br_train,[]); %COD based pseudo inverse of the branch hidden layer
    % C=((Vp*(T11\(Qr'*G))*Vp2)/T112)*Qr2'; %external weights
    toc
    disp('gpu-based computations')
end
if n_out==1
    if flag_check==0
    C=Vp*(T11\(Qr'*G))*invBr;
    else
    C=((Vp*(T11\(Qr'*G))*Vp2)/T112)*Qr2'; %external weights
    end
    if isgpuarray(C)==1
        C=gather(C);
    end
elseif n_out >1
    for jj=1:n_out
        ind=(1:Ny)+(jj-1)*Ny;
        if flag_check==0
        C{jj}=Vp*(T11\(Qr'*G(ind,:)))*invBr;
        else
        C{jj}=((Vp*(T11\(Qr'*G(ind,:)))*Vp2)/T112)*Qr2'; %external weights
        end
        if isgpuarray(C{jj})==1
            C{jj}=gather(C{jj});
        end
    end
end
%save net
net.tr_fT=tr_fT_name;
net.n_out=n_out;
net.alphat=alphat;
net.betat=betat;
net.tr_fB=tr_fB_name;
net.alphab0=alphab0;
net.alphab1=alphab1;
net.betab=betab;
net.C=C;
net.parametric=1;
net.flag_norm=flag_norm;
net.identity=identity;
end