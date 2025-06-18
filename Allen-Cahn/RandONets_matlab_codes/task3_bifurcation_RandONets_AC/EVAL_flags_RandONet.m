function G=EVAL_flags_RandONet(RandONet,ff,yy,simmetry,parametric,flag_single)
Ny=length(yy);
if simmetry==1
    if parametric==0
        if flag_single==0
        G=(eval_RandONet(RandONet,ff,yy)-...
        eval_RandONet(RandONet,[-ff(1:Ny,:);ff(Ny+1,:)],yy))/2;
        else
        G=(eval_RandONet(RandONet,ff(1:Ny,:),yy)-...
        eval_RandONet(RandONet,-ff(1:Ny,:),yy))/2;
        end
    else
        G=(eval_RandONet_parametric(RandONet,ff(1:Ny,:),ff(Ny+1,:),yy)-...
        eval_RandONet_parametric(RandONet,-ff(1:Ny,:),ff(Ny+1,:),yy))/2;
    end
else
    if parametric==0
        if flag_single==0
        G=eval_RandONet(RandONet,ff,yy);
        else
        G=eval_RandONet(RandONet,ff(1:Ny,:),yy);
        end
    else
        G=eval_RandONet_parametric(RandONet,ff(1:Ny,:),ff(Ny+1,:),yy);
    end
end
end