function G=EVAL_flags_RandONet(RandONet,ff,pp,yy,simmetry,parametric,flag_single)
%Ny=size(ff,1)-1;
if simmetry==1
    if parametric==0
        if flag_single==0
        G=(eval_RandONet(RandONet,[ff;pp],yy)-...
        eval_RandONet(RandONet,[-ff;pp],yy))/2;
        else
        G=(eval_RandONet(RandONet,ff,yy)-...
        eval_RandONet(RandONet,-ff,yy))/2;
        end
    else
        G=(eval_RandONet_parametric(RandONet,ff,pp,yy)-...
        eval_RandONet_parametric(RandONet,-ff,pp,yy))/2;
    end
else
    if parametric==0
        if flag_single==0
        G=eval_RandONet(RandONet,[ff;pp],yy);
        else
        G=eval_RandONet(RandONet,ff,yy);
        end
    else
        G=eval_RandONet_parametric(RandONet,ff,pp,yy);
    end
end
end