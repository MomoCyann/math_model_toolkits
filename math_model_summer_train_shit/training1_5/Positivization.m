
%正向化 Positivization 三个输入变量分别为目前处理的列向量 该列的指标类型 目前处理的是第几列
%输出变量为正向化后的列向量
function [posit_x]=Positivization(x,type,i)
    if type==1 %极小型
        posit_x=max(x)-x;
        %posit_x=1./x 如果该列数据全部大于0 也可以这样正向化
    elseif type==2%中间型
        best=input('请输入最佳的值:  ');
        M=max(abs(x-best));
        posit_x=1-abs(x-best)/M;
    elseif type==3%区间型
        a=input('请输入区间下限:  ');
        b=input('请输入区间上限:  ');
        MM=max(a-min(x),max(x)-b);
        posit_x = zeros(size(x,1),1);
        for i=1:size(x,1)
            if x(i)<a
                posit_x(i)=1-(a-x(i))/MM;
            elseif x(i)>b
                posit_x(i)=1-(x(i)-b)/MM;
            else
                posit_x(i)=1;
            end
        end
    else 
        disp('请正确输入指标类型')
    end
end
