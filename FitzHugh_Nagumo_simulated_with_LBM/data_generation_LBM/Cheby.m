function nodi=Cheby(a,b,n,type)
if nargin==3
    type=1;
end
if type==1 %Cheby-gauss di tipo 1
    for i=1:n
            nodi(i)=cos((2*i-1)/(2*n)*pi);
    end
elseif type==2 %Cheby-gausslobatto di tipo2
    for i=0:n-1
            nodi(i+1)=cos(i/(n-1)*pi);
    end
end
nodi=fliplr(nodi);
len=b-a;
nodi=(nodi+1)/2*len+a;
end