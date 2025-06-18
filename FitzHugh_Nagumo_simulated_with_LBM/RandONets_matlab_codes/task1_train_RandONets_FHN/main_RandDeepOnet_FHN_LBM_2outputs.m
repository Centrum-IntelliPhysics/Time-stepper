clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
%load('data_LBM_FHN_longer.mat') %dt=1
load('data_LBM_FHN_dt01.mat') %dt=0.1
%load('data_LBM_FHN_shorter.mat') %dt=0.1
%load('data_LBM_FHN_shorter2.mat') %dt=0.01
rng(5)
save_on=0;
%
ep0=min(PARAM_train_branch);
epf=max(PARAM_train_branch);
gpu_flag=1; %training with GPUs
parametric=1; %parametric DeepONet
identity=0;
% select if you want (1) also JL RandOnet or not (2)
%select how many epsilon you want
best_RandONet=[];
best_err=1e10;
best_knb=1;
flag_single=0;
simmetry=0;
%simmetry=1;
%
% val_epsilon=PARAM_test_branch(500); %0.1 or 1 (is effectively 3.8 max)  <--------------------
% %val_epsilon=1;
% %
% if val_epsilon<1
%     parametric=0;
% end
% if parametric==1
%     val_epsilon=1; %cannot do parametric with parameter fixed
% end
% if val_epsilon<1
%     I1=UV0train_branch(end,:)==val_epsilon;
%     I2=UV0test_branch(end,:)==val_epsilon;
%     flag_single=1;
% else
%     flag_single=0;
%     I1=UV0train_branch(end,:)<val_epsilon;
%     I2=UV0test_branch(end,:)<val_epsilon;
% end
% ns=sum(I1);
% G_test=Vtest_out(:,I2);
% G_train=Vtrain_out(:,I1);
% %ns=size(I1,2);
% I3=randperm(ns,round(ns*0.2));
% G_val=G_train(:,I3);
% G_train(:,I3)=[];
% %G_val=G_test;
% %
% ff_test=UV0test_branch(:,I2);
% ff_train=UV0train_branch(:,I1);
% ff_val=ff_train(:,I3);
% ff_train(:,I3)=[];
%
ss=size(UV0_train_branch,2);
II=randperm(ss,round(ss*0.2));
ff_train=UV0_train_branch(:,II);
param_train=PARAM_train_branch(:,II);
gg_train_out=[U1_train_out(:,II);V1_train_out(:,II)];
%
UV0_train_branch(:,II)=[];
PARAM_train_branch(:,II)=[];
U1_train_out(:,II)=[];
V1_train_out(:,II)=[];
%
ss1=size(UV0_train_branch,2);
II1=randperm(ss1,round(ss1*0.5));
ff_val=UV0_train_branch(:,II1);
param_val=PARAM_train_branch(:,II1);
gg_val_out=[U1_train_out(:,II1);V1_train_out(:,II1)];
%
UV0_train_branch(:,II1)=[];
PARAM_train_branch(:,II1)=[];
U1_train_out(:,II1)=[];
V1_train_out(:,II1)=[];
%
ff_test=[UV0_test_branch,UV0_train_branch];
param_test=[PARAM_test_branch,PARAM_train_branch];
gg_test_out=[U1_test_out,U1_train_out;V1_test_out,V1_train_out];

%
pod_modes=1;
if pod_modes==1
bias_proj=mean(ff_val,2);
%Cov=1/(size(ff_val,2)-1)*(ff_val-bias_proj)*(ff_val-bias_proj)';
Cov=ff_val*ff_val';
[Vkov,Dkov]=eig(Cov);
Dkov=max(real(diag(Dkov)),0);
[Dkov,Iord]=sort(Dkov,1,"descend");
Vkov=Vkov(:,Iord);
tol=0.999;
tempkov=0;
Skov=sum(Dkov);
kcov=0;
while tol>tempkov
    kcov=kcov+1;
    tempkov=tempkov+Dkov(kcov)/Skov;
end
Vkov_proj=Vkov(:,1:kcov)';
save('PODmodes2.mat','Vkov_proj');
%
ff_train=Vkov_proj*(ff_train-bias_proj);
ff_test=Vkov_proj*(ff_test-bias_proj);
ff_val=Vkov_proj*(ff_val-bias_proj);
%
%gg_test_out=Vkov_proj*gg_test_out;
%gg_train_out=Vkov_proj*gg_train_out;
%gg_val_out=Vkov_proj*gg_val_out;
end
%
dt=dtRON;
Ny=length(Ytrunk); %51; 
%
Nt=Ny*3; %300; %number of neurons in the trunk hidden layer
NNb=[10,20,40,80,100,150,300,500,1000];%1000,1500,2000];%,1500,2000]; %number of neurons in the branch hidden layer
kmodel=2;
%data_case='fews'; iters=10;
data_case='many'; iters=3; %Iterations are used solely to improve computational time estimation.
n_out=2;
%
variation_on=0;
if variation_on==1
    gg_train_out=gg_train_out-ff_train;
    gg_test_out=gg_test_out-ff_test;
    gg_val_out=gg_val_out-ff_val;
end
%
%
knb=0;
for Nb=NNb
    knb=knb+1;
    knb/length(NNb)
    pause(0.0001)
    timetime=0;
    for it=1:iters
        err_iter=1e10;
        tstart=tic;
        RandONet=train_RandONet_parametric(ff_train,param_train,Ytrunk,gg_train_out,Nt,Nb,kmodel,gpu_flag,n_out,identity);
        %
        timetime=timetime+toc(tstart);
        %
        %validation/ (or train/test set) (to select) (should be validation))
        Gnet_val=EVAL_flags_RandONet(RandONet,ff_val,param_val,Ytrunk,parametric);
        %
        %err_temp=max(max(abs(gg_val_out-Gnet_val)));
        err_temp=prctile(sqrt(sum((gg_val_out-Gnet_val).^2)),95,2); %sqrt(sum((gg_val_out-Gnet_val).^2));
        if err_iter>err_temp
            save_iter_RandONet=RandONet;
            err_iter=err_temp;
        end
            %
    end %(end iters)
    timetime=timetime/iters;
    RandONet=save_iter_RandONet;
    %
    %test set
    Gnet_test=EVAL_flags_RandONet(RandONet,ff_test,param_test,Ytrunk,parametric);
    errMSE_RFFN_test(knb)=mean(mean((gg_test_out-Gnet_test).^2));
    errmax_RFFN_test(knb)=max(max(abs(gg_test_out-Gnet_test)));
    errL2_RFFN_test(knb,:)=sqrt(sum((gg_test_out-Gnet_test).^2));
    if best_err>prctile(errL2_RFFN_test(knb,:),95,2) %errmax_RFFN_test(knb)
        best_err=prctile(errL2_RFFN_test(knb,:),95,2);
        best_RandONet=RandONet;
        best_knb=knb;
    end
    time_RFFN(knb)=timetime;
end
%
errmL2_RFFN_test=median(errL2_RFFN_test,2)';
err_95L2_RFFN_test=prctile(errL2_RFFN_test,95,2)';
err_05L2_RFFN_test=prctile(errL2_RFFN_test,5,2)';


figure(1)
hold off
fill([NNb,fliplr(NNb)],[err_05L2_RFFN_test,fliplr(err_95L2_RFFN_test)],...
    'm','FaceAlpha',0.3,'LineStyle','none')
hold 
set(gca,'Xscale','log','Yscale','log')
loglog(NNb,errmL2_RFFN_test,'x-r','MarkerSize',8)
loglog(NNb,errMSE_RFFN_test,'x--c','MarkerSize',8)
loglog(NNb,errmax_RFFN_test,'x:b','MarkerSize',8)
%loglog(NNb(best_knb),errmax_RFFN_test(best_knb),'s-g','MarkerSize',10)
loglog(NNb(best_knb),errmL2_RFFN_test(best_knb),'s-g','MarkerSize',10)
grid on
set(gca,'FontSize',18,'YTick',10.^(-32:2:32))
xlabel('$M$ (branch neurons)','Interpreter','latex')
ylabel('error')
legend('RFFN 5\%-95\% $L^2$','RFFN median $L^2$','RFFN MSE',...
    'RFFN MaxAE','interpreter','latex','NumColumns',2,'FontSize',14)
% legend('JL max-$L^2$','JL MSE','JL MaxAE',...
%     'RFFN max-$L^2$','RFFN MSE','RFFN MaxAE','interpreter','latex',...
%     'NumColumns',2,'FontSize',14)
VV=axis;
ylim(10.^[log10(VV(3))*1.1,log10(VV(4))+(log10(VV(4))-log10(VV(3)))/4])
%

%
figure(2)
hold off
fill([time_RFFN,fliplr(time_RFFN)],[err_05L2_RFFN_test,fliplr(err_95L2_RFFN_test)],...
    'm','FaceAlpha',0.3,'LineStyle','none')
hold on
set(gca,'Xscale','log','Yscale','log')
loglog(time_RFFN,errmL2_RFFN_test,'x-r','MarkerSize',8)
loglog(time_RFFN,errMSE_RFFN_test,'x--c','MarkerSize',8)
loglog(time_RFFN,errmax_RFFN_test,'x:b','MarkerSize',8)
%loglog(time_RFFN(best_knb),errmax_RFFN_test(best_knb),'s-g','MarkerSize',10)
loglog(time_RFFN(best_knb),errmL2_RFFN_test(best_knb),'s-g','MarkerSize',10)
grid on
set(gca,'FontSize',18,'YTick',10.^(-32:2:32))
xlabel('time [seconds]','Interpreter','latex')
ylabel('error')
    legend('RFFN 5\%-95\% $L^2$','RFFN median $L^2$','RFFN MSE',...
    'RFFN MaxAE','interpreter','latex','NumColumns',2,'FontSize',14)
%VV=axis;
ylim(10.^[log10(VV(3))*1.1,log10(VV(4))+(log10(VV(4))-log10(VV(3)))/4])
%
%

%
tot_ele=size(ff_test,2);

for i=1:9
    k=round((i-1)/9*tot_ele+1);%plot(repmat(Ytrunk,n_out,1),
    Gnetk=EVAL_flags_RandONet(best_RandONet,ff_test(:,k),param_test(k),Ytrunk,parametric);
    figure(3)
    subplot(3,3,i)
    hold off
    for ki=1:n_out
    plot(Ytrunk,gg_test_out((1:Ny)+Ny*(ki-1),k))
    hold on
    plot(Ytrunk,Gnetk((1:Ny)+Ny*(ki-1)),'--')
    end
    
    figure(4)
    subplot(3,3,i)
    hold off
    for ki=1:n_out
    semilogy(Ytrunk,abs(gg_test_out((1:Ny)+Ny*(ki-1),k)-Gnetk((1:Ny)+Ny*(ki-1)) ))
    hold on
    %plot(repmat(Ytrunk,n_out,1),abs(gg_test_out(:,k)-Gnetk))
    end
end
folder='figures/';
if parametric==1
fig_base='fig_FHN_022_070_multi_par_RandONet_';
else
fig_base='fig_FHN_022_single_par_RandONet_';
end
fig_name={'for_neurons','for_time','some_sols','some_errs'};
if save_on==1
    for i=1:4
        filename = fullfile(folder, [fig_base,fig_name{i}]);
        figure(i)
        pause(0.001)
        fig=gcf;
        % Save as .fig
        savefig(fig, [filename, '.fig']);
        % Save as .eps
        saveas(fig, [filename, '.eps'], 'epsc');
        % Save as .pdf
        saveas(fig, [filename, '.pdf'], 'pdf');
        % Save as .jpg
        saveas(fig, [filename, '.jpg'], 'jpg');
    end
end

%
RandONet=best_RandONet;
if save_on==1
if parametric==1
    if pod_modes==1
        save('POD_RandONet_parametric_FHN_0005_0995','RandONet','flag_single','parametric','simmetry')
    else
        save('RandONet_parametric_FHN_0005_0995','RandONet','flag_single','parametric','simmetry')
    end
else
    save('RandONet_single_FHN_022','RandONet','flag_single','parametric','simmetry')
end
if pod_modes==1
    save('PODmodes','Vkov_proj','bias_proj')
end
end
