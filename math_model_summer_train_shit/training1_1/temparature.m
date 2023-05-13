clc;clear all;
summer=[26,26,25,25,24,25,26,27,29,30,31,31,31,32,32,32,32,31,30,29,29,28,28,27,26];
sum_min=[0,60,120,180,240,300,360,420,480,540,600,660,720,780,840,900,960,1020,1080,1140,1200,1260,1320,1380,1439];
winter=[6,6,6,6,6,6,6,7,7,7,7,8,9,9,10,10,10,10,9,8,7,6,5,4,6];
win_min=[0,60,120,180,240,300,360,420,480,540,600,660,720,780,840,900,960,1020,1080,1140,1200,1260,1320,1380,1439];

x1=0:1:1439;


length(summer)

figure(1)
subplot(1,2,1)
scatter(sum_min,summer,'*')
hold
y1=interp1(sum_min,summer,x1,'cubic');
plot(x1,y1)

subplot(1,2,2)
scatter(win_min,winter)
hold
y2=interp1(win_min,winter,x1,'cubic');
plot(x1,y2)
