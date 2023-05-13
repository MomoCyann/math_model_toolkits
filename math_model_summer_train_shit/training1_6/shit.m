% 需要解决的问题：
% 1.某些工序限定用某些机器
% 可以就做26个机器，然后判定只要不大于那个最大数量，就可以同时做？
% 然后规定每个工序增加的数量x，时间也缩短x倍

% 对不齐的工序添加工期为0 的工序方便 染色体排序

% 产量怎么和订单对上？
% 先计算出根据当前速度（机器倍数），需要多少时间才能制造需要数量，把这个作为最后工期

function shit() 
    [stageMachQty,ptimes]=initProblem(); 
    machProcesses=getMachProc(stageMachQty); 
    [jobQty,procQty]=size(ptimes); 
    gaParas=initGA();
    pop=gaParas(1); 
    crossRate=gaParas(2); 
    muteRate=gaParas(3); 
    elitistQty=gaParas(4); 
    pLength=jobQty*procQty; 
    chromes=initChromes(pop,pLength);
    fit=getFitnesses(chromes,ptimes,machProcesses);
    %初始化最优解 
    [sortFit,sortIdx]=sortrows(fit); 
    optFit=sortFit(1); 
    optChrome=chromes(sortIdx(1),:);

    maxGeneration=50; 
    nowGeneration=0;
    while(maxGeneration>=nowGeneration)
        %进行遗传的迭代操作
        %交叉操作 
        crossedChromes=crossChromes(chromes,crossRate);
        %变异
        mutedChromes=muteChromesInsert(crossedChromes,muteRate);
        %选择操作 
        fit=getFitnesses(mutedChromes,ptimes,machProcesses);
        chromes=elitestSelection(fit,mutedChromes,elitistQty);
        [sortFit,sortIdx]=sortrows(fit); 
        if(sortFit(1)<=optFit)
            optFit=sortFit(1) 
            optChrome=mutedChromes(sortIdx(1),:);
        end
        nowGeneration=nowGeneration+1;
    end
    optFit
    schedule=getSchedule2(optChrome,ptimes,machProcesses)
    drawGant(schedule)
end


function outChromes=muteChromesInsert(chromes,muteRate) 
[pop,len]=size(chromes);
for i=1:pop nowChrome=chromes(i,:); 
    if rand()<muteRate
        points=randperm(len,2); 
        if points(1)<points(2)
            insertChrome=nowChrome(points(1):points(2)-1); 
            newChrome=circshift(insertChrome,-1); 
            chromes(i,points(1):points(2)-1)=newChrome;
        else
            insertChrome=nowChrome(points(2)+1:points(1)); newChrome=circshift(insertChrome,1); chromes(i,points(2)+1:points(1))=newChrome;
        end 
    end
    outChromes=chromes; 
end
end

function outChromes=crossChromes(chromes,crossRate) 
pop=size(chromes,1);
cols=size(chromes,2); 
for i=1:2:pop
    if rand()<crossRate %进行交叉操作 
        parent1=chromes(i,:); parent2=chromes(i+1,:); p=sortrows(randperm(cols,2)')'; crossP1=parent1(p(1):p(2)); crossP2=parent2(p(1):p(2)); son1=parent1; son1(p(1):p(2))=crossP2; son2=parent2; son2(p(1):p(2))=crossP1;
        %子片段对比 
        crossLen=size(crossP1,2); 
        for j=crossLen:-1:1
            midCode=crossP1(j); 
            for k=1:size(crossP2,2)
                if crossP2(k)==midCode crossP1(j)=[];
                    crossP2(k)=[]; break;
                end 
            end
        end
        %染色体编码置换 
        repeatNum=size(crossP1,2); 
        if repeatNum>0
            %对子代1的有效性置换 
            for j=1:repeatNum
                midCode=crossP2(j); 
                if p(1)==1
                    for k=p(2)+1:cols
                        if son1(k)==midCode son1(k)=crossP1(j); break;
                        end 
                    end
                else
                    if p(2)==cols 
                        for k=1:p(1)-1
                            if son1(k)==midCode son1(k)=crossP1(j); break;
                            end 
                        end
                    else
                        getIt=0;
                        for k=1:p(1)-1
                            if son1(k)==midCode son1(k)=crossP1(j); getIt=1;
                                break; end
                        end
                        if getIt==0
                            for k=p(2)+1:cols
                                if son1(k)==midCode son1(k)=crossP1(j); break;
                                end
                            end
                        end 
                    end
                end 
            end
            %对子代2的有效性置换
            for j=1:repeatNum
                midCode=crossP1(j); 
                if p(1)==1
                    for k=p(2)+1:cols
                        if son2(k)==midCode son2(k)=crossP2(j); break;
                        end 
                    end
                else
                    if p(2)==cols 
                        for k=1:p(1)-1
                            if son2(k)==midCode son2(k)=crossP2(j); break;
                            end 
                        end
                    else
                        getIt=0;
                        for k=1:p(1)-1
                            if son2(k)==midCode son2(k)=crossP2(j); getIt=1;
                                break; end
                        end
                        if getIt==0
                            for k=p(2)+1:cols
                                if son2(k)==midCode son2(k)=crossP2(j); break;
                                end 
                            end
                        end 
                    end
                end 
            end
        end
        chromes(i,:)=son1; chromes(i+1,:)=son2;
    end 
end
outChromes=chromes; 
end

function selectedChromosomes=elitestSelection(fitnessValue,mutedChromosomes,elitistQty)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	selection by elisist retained strategy	%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[population,geneLth]=size(mutedChromosomes); [~,queueIndex]=sort(fitnessValue);
%get the elitest chromosomes 
elitist=zeros(elitistQty,geneLth); 
for num=1:elitistQty
    elitist(num,:)=mutedChromosomes(queueIndex(num),:);
end
minFit=min(fitnessValue); 
if minFit<0
    fitnessValue=fitnessValue+abs(minFit);
end 
maxFit=max(fitnessValue)*1.2; fitnessValue=maxFit-fitnessValue; sumFitness=sum(fitnessValue); accumFitness=zeros(1,population); accumFitness(1)=fitnessValue(1);
%select chromosomes by roullette method 
for i=2:population
    accumFitness(i)=accumFitness(i-1)+fitnessValue(i);
end
selectRate=unifrnd(0,sumFitness,[1 population]); 
selectedChromosomes=zeros(population,geneLth);
for i=1:population 
    selectNumber=find(selectRate(i)<=accumFitness); 
    selectedChromosomes(i,:)=mutedChromosomes(selectNumber(1),:);
end
%integrate the elistists into son chromosomes 
selectedChromosomes(1:elitistQty,:)=elitist; randIdx=randperm(population)'; selectedChromosomes=selectedChromosomes(randIdx,:);
end

function fitnesses=getFitnesses(chromes,ptimes,machProcesses) 
[pop,~]=size(chromes);
fitnesses=zeros(pop,1); 
for i=1:pop
    schedule=getSchedule2(chromes(i,:),ptimes,machProcesses); cmax=max(schedule(:,5));
    fitnesses(i)=cmax;
end 
end

%Step4的小程序：根据stageMachQty数组生成工序和机器编码之间的二维数组 
function machProcesses=getMachProc(stageMachQty)
procQty=max(size(stageMachQty)); cols=sum(stageMachQty); machProcesses=zeros(2,cols); midIdx=1;
for i=1:procQty 
    myMachs=stageMachQty(i); 
    for j=1:myMachs
        machProcesses(2,midIdx)=i; machProcesses(1,midIdx)=midIdx; midIdx=midIdx+1;
    end
end
end

%Step1 问题相关数据初始化
function [stageMachQty,ptimes]=initProblem() 
stageMachQty=[1	1	1	1	1	1	1	1	1	1	1	2	2	1	1	1	1	1	1	1	1	1	9
];
ptimes=[7	12	3.25000000000000	4.50000000000000	1	17.8750000000000	7.15000000000000	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	29.7500000000000	0	0	0	0	0	0	0	0	0	0	0	0	0	0	11.9000000000000
0	0	0	0	0	0	0	0	6.96000000000000	4.50000000000000	4	2	28.7000000000000	0	0	0	0	0	0	0	0	0	4.25000000000000
0	0	0	0	0	0	0	0	0	0	0	0	30	0	0	0	0	0	0	0	0	0	17.8000000000000
0	0	0	0	0	0	0	0	0	0	0	0	0	3.75000000000000	18.7500000000000	0	0	0	0	0	0	0	15
0	0	0	0	0	0	0	0	0	0	0	2	0	0	0	3.70000000000000	5	6	20	0	0	0	12
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	36.4500000000000	0	0	18.2250000000000
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	33.4000000000000	0	3.75000000000000
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	30.9000000000000	15.4500000000000];					
						

end

function gaParas=initGA()
pop=60; crossRate=0.6; muteRate=0.4; elistQty=5;
gaParas=[pop,crossRate,muteRate,elistQty]; end

function chromes=initChromes(pop,pLength) 
chromes=zeros(pop,pLength);
for i=1:pop
chromes(i,:)=randperm(pLength);
end
end

function schedule=getSchedule2(singleChrome,ptimes,machProcArray) 
    bigM=1000000;
    [jobQty,procQty]=size(ptimes); 
    jobPreEndTime=zeros(1,jobQty); 
    machQty=size(machProcArray,2);
    chromeLength=max(size(singleChrome));
    schedule=zeros(0,5);
    for i=1:procQty
    %首先获取当前工序的可用机器编码 
    nowMachIds=machProcArray(1,machProcArray(2,:)==i); 
    nowMachQty=size(nowMachIds,2); 
    midIdx=singleChrome<=jobQty*i; 
    midChrome=singleChrome(midIdx); 
    midIdx=midChrome>jobQty*(i-1); 
    midChrome=midChrome(midIdx); 
    jobSeq=mod((midChrome+1),jobQty)+1;
    %依次为jobSeq中的作业安排当前的机器及其开工、完工时间 
    for j=1:jobQty
        nowJobId=jobSeq(j);
        %获取这些可用机器中最早可用的机器编码 
        nowSchedule=zeros(0,5);

        %获取当前工序各台机器已经安排的调度方案
    
        for k=1:nowMachQty 
            midMachId=nowMachIds(k);
            midSch=schedule(schedule(:,2)==midMachId,:); 
            nowSchedule=[nowSchedule;midSch];
        end
        %获取当前工序各台机器可用时段 
        emptySlot=zeros(0,3);
        nowSchedule=sortrows(nowSchedule,[2,4]);%先按照机器编码和开工时间升序排序
        for k=1:nowMachQty
            midMachId=nowMachIds(k); 
            midSch=nowSchedule(nowSchedule(:,2)==midMachId,:); 
            myQty=size(midSch,1);
            if(myQty==0)
                midSlot=[midMachId 0 bigM]; emptySlot=[emptySlot;midSlot];
            else
                preEmptyTime=0; 
                for kk=1:myQty
                    midSlot=[midMachId preEmptyTime midSch(kk,4)]; 
                    emptySlot=[emptySlot;midSlot]; 
                    preEmptyTime=midSch(kk,5);
                end
                midSlot=[midMachId preEmptyTime bigM]; 
                emptySlot=[emptySlot;midSlot];
            end
        end
        %对emptySlot进行排序，按照slot的开始时间升序排序 
        emptySlot=sortrows(emptySlot,2);
        emptyQty=size(emptySlot,1);
        %依次判断emptySlot中的空闲间距是否能够放下当前操作 
        jCanStartTime=jobPreEndTime(nowJobId); 
        jobPTime=ptimes(nowJobId,i);
        for kk=1:emptyQty 
            midCanStartTime=max(jCanStartTime,emptySlot(kk,2)); 
            if (emptySlot(kk,3)-midCanStartTime>=jobPTime)
                myEndTime=midCanStartTime+jobPTime;
                midSch=[nowJobId emptySlot(kk,1) i midCanStartTime myEndTime]; 
                schedule=[schedule;midSch];  jobPreEndTime(nowJobId)=myEndTime;
                break;
            end
        end
    end
    end 
end

%根据调度方案绘制甘特图程序
%schedule:1-jobId,2-machId,3-procId,4-startTime,5-endTime
function drawGant(schedule)
rows=size(schedule,1);
maxMachId=max(schedule(:,2)); 
jobQty=max(schedule(:,1));
%mycolor=rand(jobQty,3); 
mycolor = ['#ff0000';'#ffed00';'#ff0092';'#c2ff00';'#00c7f2';'#c1f1fc';'#ebffac';'#ffc2e5';'#ffaaaa';];
figure;
ylim([0 maxMachId+1]); 
for i=1:rows
x=schedule(i,4:5); y=[schedule(i,2) schedule(i,2)];
line(x,y,'lineWidth',16,'color',mycolor(schedule(i,1),:)); procId=schedule(i,3);
jobId=schedule(i,1);
if x(1)~=x(2) && x(2)~=0
txt=['[' int2str(jobId) ' ' int2str(procId) ']']; text(mean(x)-1,y(1),txt);
end 
end
end

 


