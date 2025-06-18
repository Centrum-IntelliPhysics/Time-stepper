clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
rng(6)
load('data_00_38_Bratu_patches_chebfun.mat')
%
%
%1 is left derivative (or more point)
%2:Np_teeth+1 points
%Np_gaps+2 is right derivative (or more point)
%Np_gaps+3 is location of mid grid point! (non-local operator, or local but inhomogenous)
%parameter is external saved
%
save_on=1; %<----------------
%
gpu_flag=1; %training with GPUs  <----------
parametric=1; %parametric DeepONet <----------
% select if you want (1) also JL RandOnet or not (2)
flag_JL=2; %2 is not, 1 is yes <--------------
if flag_JL~=1
    flag_JL=2;
end
%select how many epsilon you want
% eps=0.22 or +0.02 up to 0.7, 1 for all
val_epsilon=1; %1 or 3.9 (is effectively 3.8 max)  <--------------------
%
if val_epsilon<3.8
    parametric=0;
end
if parametric==1
    val_epsilon=3.9; %cannot do parametric with parameter fixed
end
if val_epsilon<=3.8
    I1=param_train_branch==val_epsilon;
    I2=param_test_branch==val_epsilon;
    I3=param_val_branch==val_epsilon;
    flag_single=1;
else
    flag_single=0;
    I1=param_train_branch<val_epsilon;
    I2=param_test_branch<val_epsilon;
    I3=param_val_branch<val_epsilon;
end
ns=sum(I1);

if ns>1e5
    I1=find(I1);
    I11=randperm(ns,1e5);
    I1=I1(I11);
end
G_train=Vtrain_out(:,I1);
G_test=Vtest_out(:,I2);
G_val=Vval_out(:,I3);
%
ff_train=Utrain_branch(:,I1);
ff_train=ff_train+randn(size(ff_train))*1e-6;
p_train=param_train_branch(1,I1);
ff_test=Utest_branch(:,I2);
p_test=param_test_branch(1,I2);
ff_val=Uval_branch(:,I3);
p_val=param_val_branch(1,I3);

%

yy=Ytrain_trunk;
%
%dt=0.001;
variation_on=0; Ny=11; %variation is basically if one find the map Un+1=Un+f(Un) (so the difference)
simmetry=0;  %if 1 then compute (f(u)+f(-u))/2 <---------------------
if variation_on==1
    G_train=(G_train-ff_train);
    G_test=(G_test-ff_test);
end
%
Nt=max(size(Utrain_branch,1)*3,100); %number of neurons in the trunk hidden layer
NNb=[10,20,40,80,100,150,300,500,1000];%,750,1000,1500,2000];%,500,1000,2000]; %number of neurons in the branch hidden layer
%NNb=[10,100,300,2500];
%data_case='fews'; iters=10;
data_case='many';
iters=3; % <-------------------
%Iterations are used solely to improve computational time estimation.
%
knb=0;
%
best_RandONet=[];
best_err=1e10;
best_knb=1;
for Nb=NNb
    knb=knb+1;
    knb/length(NNb)
    for kmodel=flag_JL:2
        pause(0.0001)
        if kmodel==1
            tr_fB=@(x) x; flag_branch=1; %%%%%%%JL
        elseif kmodel==2
            tr_fB=@(x) cos(x); flag_branch=0;  %%%%%%RFFN 
        end
        timetime=zeros(1,iters);
        err_iter=1e10;
        for it=1:iters
            tstart=tic;
            if parametric==0
                if flag_single==0
                    RandONet=train_RandONet([ff_train;p_train],yy,G_train,Nt,Nb,kmodel,gpu_flag);
                else
                    RandONet=train_RandONet(ff_train,yy,G_train,Nt,Nb,kmodel,gpu_flag);
                end
            else
                RandONet=train_RandONet_parametric(ff_train,p_train,yy,G_train,Nt,Nb,kmodel,gpu_flag);
            end
            timetime(it)=toc(tstart);
            %
            %validation/ (or train/test set) (to select) (should be validation))
            Gnet_val=EVAL_flags_RandONet(RandONet,ff_val,p_val,yy,simmetry,parametric,flag_single);
            %
            err_temp=max(max(abs(G_val-Gnet_val)));
            if err_iter>err_temp
                save_iter_RandONet=RandONet;
                err_iter=err_temp;
                fprintf('new best, err=%2.4e  ######### \n',err_iter)
            end
            %
        end %(end iters)
        timetime=median(timetime);
        RandONet=save_iter_RandONet;
        %
        %test set
        Gnet_test=EVAL_flags_RandONet(RandONet,ff_test,p_test,yy,simmetry,parametric,flag_single);
        if kmodel==1
            errMSE_JL_test(knb)=mean(mean((G_test-Gnet_test).^2));
            errmax_JL_test(knb)=max(max(abs(G_test-Gnet_test)));
            errL2_JL_test(knb,:)=sqrt(sum((G_test-Gnet_test).^2));
            time_JL(knb)=timetime;
        elseif kmodel==2
            errMSE_RFFN_test(knb)=mean(mean((G_test-Gnet_test).^2));
            errmax_RFFN_test(knb)=max(max(abs(G_test-Gnet_test)));
            errL2_RFFN_test(knb,:)=sqrt(sum((G_test-Gnet_test).^2));
            if best_err>errmax_RFFN_test(knb)
                best_err=errmax_RFFN_test(knb);
                best_RandONet=RandONet;
                best_knb=knb;
            end
            time_RFFN(knb)=timetime;
        end
    end
%
end
errmL2_RFFN_test=median(errL2_RFFN_test,2)';
err95L2_RFFN_test=prctile(errL2_RFFN_test,95,2)';
err05L2_RFFN_test=prctile(errL2_RFFN_test,5,2)';
if flag_JL==1
    errmL2_JL_test=median(errL2_JL_test,2)';
    err95L2_JL_test=prctile(errL2_JL_test,95,2)';
    err05L2_JL_test=prctile(errL2_JL_test,5,2)';
end

figure(1)
hold off
if flag_JL==1
    fill([NNb,fliplr(NNb)],[err05L2_JL_test,fliplr(err95L2_JL_test)],...
        'c','FaceAlpha',0.3,'LineStyle','none')
    hold on
    loglog(NNb,errmL2_JL_test,'o-b')
    loglog(NNb,errMSE_JL_test,'o--b')
    hold on
end
fill([NNb,fliplr(NNb)],[err05L2_RFFN_test,fliplr(err95L2_RFFN_test)],...
    'm','FaceAlpha',0.3,'LineStyle','none')
hold on
loglog(NNb,errmL2_RFFN_test,'x-r','MarkerSize',8)
set(gca,'Xscale','log','Yscale','log')
loglog(NNb,errMSE_RFFN_test,'x--c','MarkerSize',8)
loglog(NNb,errmax_RFFN_test,'x:b','MarkerSize',8)
loglog(NNb(best_knb),errmax_RFFN_test(best_knb),'s-g','MarkerSize',10)
grid on
set(gca,'FontSize',18,'YTick',10.^(-32:2:32))
xlabel('$M$ (branch neurons)','Interpreter','latex')
ylabel('error')
if flag_JL==1
legend('JL 5\%-95\% $L^2$','JL median $L^2$','JL MSE','RFFN 5\%-95\% $L^2$','RFFN median $L^2$','RFFN MSE','interpreter','latex','NumColumns',2,'FontSize',14)
else
legend('RFFN 5\%-95\% $L^2$','RFFN median $L^2$','RFFN MSE',...
    'RFFN MaxAE','interpreter','latex','NumColumns',2,'FontSize',14)
end
% legend('JL max-$L^2$','JL MSE','JL MaxAE',...
%     'RFFN max-$L^2$','RFFN MSE','RFFN MaxAE','interpreter','latex',...
%     'NumColumns',2,'FontSize',14)
VV=axis;
ylim(10.^[log10(VV(3))*1.1,log10(VV(4))+(log10(VV(4))-log10(VV(3)))/4])

figure(2)
hold off
if flag_JL==1
    fill([time_JL,fliplr(time_JL)],[err05L2_JL_test,fliplr(err95L2_JL_test)],...
        'c','FaceAlpha',0.3,'LineStyle','none');
    set(gca,'Xscale','log','Yscale','log')
    hold on
    loglog(time_JL,errmL2_JL_test,'o-b')
    loglog(time_JL,errMSE_JL_test,'o--b')
    hold on
end
fill([time_RFFN,fliplr(time_RFFN)],[err05L2_RFFN_test,fliplr(err95L2_RFFN_test)],...
    'm','FaceAlpha',0.3,'LineStyle','none')
hold on
set(gca,'Xscale','log','Yscale','log')
loglog(time_RFFN,errmL2_RFFN_test,'x-r','MarkerSize',8)
loglog(time_RFFN,errMSE_RFFN_test,'x--c','MarkerSize',8)
loglog(time_RFFN,errmax_RFFN_test,'x:b','MarkerSize',8)
loglog(time_RFFN(best_knb),errmax_RFFN_test(best_knb),'s-g','MarkerSize',10)
grid on
set(gca,'FontSize',18,'YTick',10.^(-32:2:32))
xlabel('time [seconds]','Interpreter','latex')
ylabel('error')
if flag_JL==1
    legend('JL 5\%-95\% $L^2$','JL median $L^2$','JL MSE',...
        'RFFN 5\%-95\% $L^2$',...
        'RFFN median $L^2$',...
        'RFFN MSE',...
        'interpreter','latex',...
        'NumColumns',2,'FontSize',14)
else
    legend('RFFN 5\%-95\% $L^2$','RFFN median $L^2$','RFFN MSE',...
    'RFFN MaxAE','interpreter','latex','NumColumns',2,'FontSize',14)
end
%VV=axis;
ylim(10.^[log10(VV(3))*1.1,log10(VV(4))+(log10(VV(4))-log10(VV(3)))/4])
%
tot_ele=size(ff_test,2);

for i=1:9
    k=round((i-1)/9*tot_ele+1);
    Gnetk=EVAL_flags_RandONet(best_RandONet,ff_test(:,k),p_test(1,k),yy,simmetry,parametric,flag_single);
    figure(3)
    subplot(3,3,i)
    hold off
    plot(yy,G_test(:,k))
    hold on
    plot(yy,Gnetk,'--')
    figure(4)
    subplot(3,3,i)
    hold off
    plot(yy,abs(G_test(:,k)-Gnetk))
end
folder='figures/';
if parametric==1
fig_base='fig_Bratu_patches_00_38_multi_par_RandONet_';
else
fig_base='fig_Bratu_patches_10_single_par_RandONet_';
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
    save('RandONet_parametric_Bratu_patches_00_38','RandONet','flag_single','parametric','simmetry')
else
    save('RandONet_single_Bratu_patches_10','RandONet','flag_single','parametric','simmetry')
end
end