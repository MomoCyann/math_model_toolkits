X = Y;
[n,m] = size(X);
disp(['共有' num2str(n) '个评价对象 共有' num2str(m) '个评价指标'])
judge=input(['这' num2str(m) '个指标是否需要正向化处理，需要请输入1 不需要请输入0：  ']);
if judge==1
    Position=input('请输入需要正向化处理的列 比如2，3，6列需要处理 则输入[2,3,6]：  ');
    disp('请输入这些列分别是什么指标类型（1：极小型 2：中间型 3：区间型）')
    Type=input('比如 2 3 6列分别是极小型 区间型 中间型 则输入[1,3,2]：  ');
    for i=1:size(Position,2)
        X(:,Position(i))=Positivization(X(:,Position(i)),Type(i),Position(i));
    end
    disp('正向化后的矩阵为 X=');
    disp(X);
end
%标准化
Z = X ./ repmat(sum(X.*X) .^ 0.5, n, 1);
%Z = X;
disp('标准化矩阵 Z = ')
disp(Z)
 
 
 
disp("请输入是否需要增加权重向量，需要输入1，不需要输入0")
Judge = input('请输入是否需要增加权重： ');
if Judge == 1
    if sum(sum(Z<0))>0
        disp('标准化矩阵中存在负数 正在重新标准化')
        for j=1:m
            minn=min(Z(:,j));
            maxx=max(Z(:,j));
            for i=1:n
                Z(i,j)=(Z(i,j)-minn)/(maxx-minn)
            end
        end
        disp('标准化完成 矩阵Z=  ');
        disp(Z);
    end
    W = Entropy_Method(Z);
    disp('熵权法确定的权重为：');
    disp(W);        
else
    W = ones(1,m) ./ m ; %如果不需要加权重就默认权重都相同，即都为1/m
end
 
 
 
D_P = sum([W .* (Z - repmat(max(Z),n,1)) .^ 2 ],2) .^ 0.5;%最优距离
D_N = sum([W .* (Z - repmat(min(Z),n,1)) .^ 2 ],2) .^ 0.5;%最劣距离
S = D_N ./ (D_P+D_N);%相对接近度（可用来当得分）   
disp('最后的得分为：')
stand_S = S / sum(S)%得分归一化 最后各方案得分相加为1
[sorted_S,index] = sort(stand_S ,'descend');
disp('按得分从高到底排列方案 分别为:  ');
disp(index+2014);%方案排名