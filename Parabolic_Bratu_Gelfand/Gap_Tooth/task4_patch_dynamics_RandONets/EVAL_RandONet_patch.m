function vv=EVAL_RandONet_patch(RandONet,uu,param,...
    ypatch,xpatches,xbordpatches,xindbord,xspan,...
    simmetry,parametric,flag_single,...
    DXpatch,dx,Nx,Nteeth,Ngaps,Np_teeth,Np_gaps)

%uu is values at xpatches
%BC=[0,0];
%spp=spline([xspan(1);xpatches(xindbord);xspan(end)],[0;uu(xindbord);0]);
%spp=spline(xpatches,uu);
%sp_der1=fnder(spp,1);
%der1_spline=ppval(sp_der1,xbordpatches);
%der1_spline=ppval(spp,xbordpatches);
%der1_spline(1)=0;
%der1_spline(end)=0;
%xspanmore=linspace(xspan(1),xspan(end),Nx*4+1);

%
vv=zeros(size(uu));
% figure(35)
% hold off
for i=1:Nteeth
    ind=(1:Np_teeth)+(i-1)*Np_teeth;
    indp=ind-Np_teeth;
    indn=ind+Np_teeth;
    %ind2=(1:Np_teeth)+(i-1)*Np_teeth+(i-1)*(Np_gaps-2);
    %indb=(1:2)+2*(i-1);
    if i>1
        ul=mean(uu(indp));
    else
        ul=0;
    end
    if i<Nteeth
        ur=mean(uu(indn));
    else
        ur=0;
    end
    Ubranch=[ul;uu(ind);ur;xpatches(ind(round(Np_teeth/2)))];
    vv(ind)=EVAL_flags_RandONet(RandONet,Ubranch,param,ypatch,simmetry,parametric,flag_single);
    % figure(35)
    % plot(xpatches(ind),uu(ind),'xb')
    % hold on
    % plot(xbordpatches(indb),der1_spline(indb),'or')
    % plot(xspan(ind2),uu(ind),'sm')
    % plot(xpatches(ind(round(Np_teeth/2))),...
    %     uu(ind(round(Np_teeth/2))),'pg')
    
end
deltal=vv(1);
deltar=vv(end);
vv(1:Np_teeth)=vv(1:Np_teeth)-deltal*(1-ypatch/ypatch(end)).^4;
vv(end-Np_teeth+1:end)=vv(end-Np_teeth+1:end)-deltar*(ypatch/ypatch(end)).^4;

% figure(33)
% hold off
% plot(xspan,spline(xpatches,uu,xspan),'-')
% hold on
% plot(xspanmore,spline(xpatches,uu,xspanmore),'--')
% plot(xpatches,uu,'s')
% plot(xpatches,vv,'.g')
%figure(34)
%hold off
%plot(xbordpatches,der1_spline,'x')
end