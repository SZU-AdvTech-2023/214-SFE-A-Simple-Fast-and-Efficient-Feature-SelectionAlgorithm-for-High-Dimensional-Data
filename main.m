


function [Mean_Cost, Mean_CA, Std_Dev, AN_SF, Worst_CA, Best_CA] = main(Max_FEs, Max_Run, g1, X, mm, su)

g2 = g1;
Input = X(:, 2:end);
Input1 = Input;
Target = X(:, 1);
featNum = size(Input, 2);
Cost = zeros(Max_FEs, Max_Run);
FN = zeros(Max_Run,1);
Run = 1;
UR = 0.3;
UR_Max = 0.3;
UR_Min = 0.001;

% %求每组特征SU的平均值
% avg_su = zeros[1, mm];
% sum = 0;
% for i = 1 : mm
%     for j = 1 : size(g{i})
%         sum = sum + su(g1{i}(j));
%     end
%     avg_su[1, i] = sum / size(g{i});
% end


while Run <= Max_Run

    EFs = 0;
    X = randi([0,1], 1, mm);      % Initialize an Individual X
    x = group2x(g1, X, featNum);
    Fit_X = Fit(Input, Target, x);           % Calculate the Fitness of X

    while EFs <= Max_FEs

        EFs = EFs + 1;
        X_New = X;


        % 1 to 0 operation
        [~, index1] = find(X == 1);
        num1 = size(index1, 2);
        k = randperm(20, 1);
        NR = ceil(rand * featNum / k);
        K1 = randi(num1, NR, 1);           
        K = index1(K1)'; 
        X_New(K) = 0;
        
        
        % 0 to 1 operation  
        if sum(X_New) == 0 
            [~, index0] = find(X == 0);
            num0 = size(index0, 2);
            k = randperm(20, 1);
            SN = ceil(rand * featNum / k);
            K1 = randi(num0, SN, 1);           
            K = index0(K1)'; 
            X_New(K) = 1;
        end
        
        x_new = group2x(g1, X_New, featNum);
        Fit_X_New = Fit(Input, Target, x_new); % Calculate the Fitness of X_New

        if Fit_X_New >= Fit_X
            X = X_New;
            Fit_X = Fit_X_New;

        end


        Cost(EFs, Run) = Fit_X;

        UR = (UR_Max - UR_Min) * ((Max_FEs - EFs) / Max_FEs) + UR_Min;  % Eq(3)
 
        disp(['SFE-CSO :   '  'Function Evaluation: ' num2str(EFs) '   Accuracy = ' num2str(Fit_X) '   Number of Selected Features = ' num2str(sum(X)) '   Run: ' num2str(Run)]);




        if EFs > 2000  && (Cost(EFs - 1, Run) - Cost(EFs - 1000, Run) == 0)


            [~, column1] = find(X == 0);
            A = [];
            for i = 1 : size(column1, 2)
                g1_data = g1{column1(1, i)};
                if isnumeric(g1_data)
                    A = [A, g1_data];
                end
            end
            Input(:, A) = [];        % S=removing unselected features from the dataset
            NP = 20;

            % Call EC for feature selection in new subspace,  S=(s1,s2,s3,�,l)

            % Call PSO for Search in the new Search Space
            [Cost, FN] = PSO1(NP, Max_FEs, EFs, Run, Input, Target, Cost, FN);


            EFs = Max_FEs + 1;
            Input = Input1;
            g1 = g2;



        end



    end




    Run = Run + 1;

end

Cost = Cost(1 : Max_FEs, :);


Mean_Cost = mean(Cost, 2);
Mean_CA = mean(Cost(end, :));
Std_Dev = std(Cost(end, :));
AN_SF = mean(FN);
Worst_CA = min(Cost(end, :));
Best_CA = max(Cost(end, :));
end


function x = group2x(g, X, featNum)
    x = zeros(1, featNum);
    for i = 1 : size(g, 1)
        x(1, g{i}) = X(1, i);
    end
end





