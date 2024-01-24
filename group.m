function [g1, mm] = group(X, f, su)

d = size(X, 2);
featNum = d - 1;

F{featNum} = {};

%%%----------对所有特征进行相关性降序排列
[value1, index1] = sort(su, 'descend');


f_feature = zeros(1, featNum);
f = 1 : featNum;

for i = 1 : featNum
    f_feature(i) = f(index1(i));
end
su1 = value1; 

for i = 2 : d
    F{i - 1} = X(:, i)';
end

m = 1;%  m表示特征组的个数
g = cell(featNum, 1);

T_su = log(featNum) * abs(su1(1) - su1(featNum)) / (featNum);  % SU差值的阈值 

% Max_mic = 0;
% for i = 1 : featNum
%     for j = (i + 1) : featNum
%         mic = mine(F{i}, F{j}).mic;
%         if mic > Max_mic
%             Max_mic = mic;
%         end
%     end
% end
T_mic = log(featNum) * 0.5 / (featNum);  % MIC差值的阈值

     
while(length(f_feature) > 1)
    f1 = [];
    i = 2;
    z = f_feature(length(f_feature));
    
    while(f_feature ~= z)
        d_su = su1(1) - su1(i);
        d_mic = mine(F{f_feature(1)}, F{f_feature(i)}).mic;
        if d_su <= T_su && d_mic >= T_mic
            f1 = [f1, f_feature(i)]; 
            f_feature(i) = []; 
            su1(i) = []; 
        else
            i = i + 1;
        end
    end

    if (su1(1) - su1(i)) <= T_su && mine(F{f_feature(1)}, F{f_feature(i)}).mic >= T_mic %%%对最后一个特征进行判断
        f1 = [f1, f_feature(i)]; 
        f_feature(i) = []; 
        su1(i) = [];
    end
    f1 = [f_feature(1), f1]; 
    f_feature(1) = []; 
    su1(1) = [];
    g{m} = f1;
    m = m + 1;
end


if length(f_feature) == 1    %%% s经过一系列的划分后，如果最后还余下一个特征，不再进行相关性判断来划分，直接放在一个组
    f1 = []; 
    f1 = [f1, f_feature(1)];
    f_feature(1) = [];
    su1(1) = []; 
    g{m} = f1;
    m = m + 1;
end

mm = m-1; %分组结束后共分了mm组

if size(g, 1) > mm
    g1 = cell(mm, 1);
    for i = 1 : mm
        g1{i} = g{i};
    end
end



      
     