function G=EVAL_flags_RandONet(RandONet,ff,param,yy,parametric)
%Ny=length(yy);
%Nx=Ny*n_out;
if parametric==0
    G=eval_RandONet(RandONet,ff,yy);
else
    G=eval_RandONet_parametric(RandONet,ff,param,yy);
end
end