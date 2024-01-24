
function [ACC] = Fit(Input, Target, X)




if sum(X)==0
    X = randi([0,1], 1, size(Input,2));
    X(1,1)=1;
    
end


[m, ~] = size(Input);
Yhat = zeros(m, 1);

Y=Target;

fold =5;
groups = 1 + rem(0 : m - 1, fold);
for group = 1 : fold
    Test_Ind = (groups == group);
    Train_Ind = ~Test_Ind;
    
    Mdl=fitcknn(Input(Train_Ind, X(1,:)==1), Target(Train_Ind, 1));
    
    
    Yhat(Test_Ind) = predict(Mdl,Input(Test_Ind, X(1,:)==1));
    Pre{group,1} = Yhat(Test_Ind);
    Pre{group,2} = Y(Test_Ind);
end

for i=1:fold
    Accururacy(i) = length(find(Pre{i,1} == Pre{i,2}))/length(Pre{i,2})*100;
end

ACC = mean(Accururacy);


end