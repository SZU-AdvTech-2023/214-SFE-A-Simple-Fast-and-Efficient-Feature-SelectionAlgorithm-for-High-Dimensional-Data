clear all
close all
clc;

addpath('C:\Users\zm\Desktop\Method\data');
load('Colon.mat');
addpath('C:\Users\zm\Desktop\Method\minepy-1.2.6\matlab');
% 保存路径
savepath;
fea = data(:, 2 : end);
label = data(:, 1);

fea = (fea - min(fea, [], 1)) ./ (max(fea, [], 1) - min(fea, [], 1));
X(:, 1) = label;
for i = 1 : size(fea, 2)
    X(:, i + 1) = fea(:, i);
end
X(isnan(X)) = 0;


tic;
%----------------------------------------------计算特征的SU
[f, su] = Cal_SU(X);

%--------------------------------------特征分组
[g1, mm] = group(X, f, su);


Max_FEs = 6000;
Max_Run = 30;


[Mean_Cost2, Mean_CA2, Std_Dev2, AN_SF2, Worst_CA2, Best_CA2] = main(Max_FEs, Max_Run, g1, X, mm);
Method1_Time = toc;



disp('*******************Method1**************************');

disp(['Worst_CA : ' num2str(Worst_CA2)]);
disp(['Best_CA  : ' num2str(Best_CA2)]);
disp(['Mean_CA  : ' num2str(Mean_CA2)]);
disp(['Std_Dev  : ' num2str(Std_Dev2)]);
disp(['AN_SF    : ' num2str(AN_SF2)]);
disp(['Time     : ' num2str(Method1_Time)]);




figure(1);


plot(Mean_Cost2,'-^','LineWidth',0.4,...
    'MarkerEdgeColor','#0000FF',...
    'MarkerFaceColor','#FFFFFF',...
    'Color' , '#0000FF',...
    'MarkerSize',4,'MarkerIndices',1:500:size(Mean_Cost2,1))

xlabel('Number of Fitness Evaluations');
ylabel('Classification Accuracy');

legend('Method1(UR=0.3)');



