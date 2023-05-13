%计算权重
function [W] = Entropy_Method(Z)
    [n,m]=size(Z);
    d=zeros(1,m);
    for i=1:m
        x = Z(:,i);
        p = x./sum(x);%概率矩阵
        e = -sum(p .* mylog(p)) / mylog(n);%信息熵
        d(i)=1-e;%信息效用值
    end
    W=d./sum(d);%熵权
end
