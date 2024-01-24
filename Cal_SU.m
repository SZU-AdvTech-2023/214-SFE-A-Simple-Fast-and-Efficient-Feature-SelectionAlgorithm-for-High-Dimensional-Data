function [f, su] = Cal_SU(X)

d = size(X, 2);
featNum = d - 1;
f = [];
su = [];
Y = X(:, 1);

for i = 1 : featNum
    b = SU(X(:,i+1), Y);
    f = [f, i];
    su = [su, b];  % c中存放这些特征对应的SU值
end

l_f = length(f);
disp(['the num of features is: ' num2str(l_f)])


su_max = max(su);
su_min = min(su);
su = (su-su_min)./(su_max-su_min);

