%% 王俊皓 2014101027 神经网络作业

clear;  % 清理数据
clc;    % 清屏

%% 输入数据和部分参数

fprintf('---- 第 1 阶段: 载入数据 ----\n\n');

% 读入数据
[X, y, y_raw, labels] = loadData('iris.txt', ' ');

% 定义层数, 如3层, 分别有2,2,3个神经元可定义为[2,2,3]
num_input = size(X, 2);
num_output = size(y, 2);
hidden_num_list = [3, 3]; % 可修改隐藏层的层数及每层个数
% 用一个数组表示神经网络的结构
unit_num_list = [num_input, hidden_num_list, num_output];

% 学习速率
ALPHA = 0.1;

% 迭代次数
ITERATION = 1000;

% 测试集比例
test_ptage = 0.2;


%% 参数初始化

% 样本数量
m = size(X, 1);
% 训练集数量
m_train = ceil(m * (1 - test_ptage));
% 测试集数量
m_test = m - m_train;

% 打乱样本顺序
sel = randperm(m);
X = X(sel,:); y = y(sel,:);

% 根据比例生成 训练集 和 测试集
Xtrain = X(1:m_train, :); ytrain = y(1:m_train, :); 
Xtest = X(m_train+1:end, :); ytest = y(m_train+1:end, :);

% raw用1,2,3表示, 而不是010, 100, 001
ytrain_raw = [];
for i = 1:m_train
    ytrain_raw = [ytrain_raw; find(ytrain(i, :), 1)];
end
ytest_raw = [];
for i = 1:m_test
    ytest_raw = [ytest_raw; find(ytest(i, :), 1)];
end

fprintf('神经网络层级结构: (输入层 -> 隐藏层 -> 输出层)\n');
fprintf('%d', unit_num_list(1));
for i = 2:size(unit_num_list, 2)
    fprintf(' -> %d', unit_num_list(i));
end
fprintf('\n\n');

fprintf('迭代次数: %d 次   学习速率: %.4f\n\n', ITERATION, ALPHA);


%% 训练

fprintf('---- 第 2 阶段: 训练模型 ----\n\n');

fprintf('<按任意键开始训练>\n')

pause;

[WEIGHT THETA ERROR] = trainingNN(Xtrain, ytrain, unit_num_list, ALPHA, ITERATION);

% 误差分析
fprintf('\n误差: (平方和)\n%.5f (第1次迭代) -> %.5f (第%d次迭代)\n\n', sum(ERROR(1,:) .^ 2), sum(ERROR(end,:) .^ 2), ITERATION);

ERROR_X = 1:ITERATION;
ERROR_Y = sum(ERROR .^ 2, 2);

plot(ERROR_X, ERROR_Y, '-');
xlabel('迭代次数');
ylabel('误差');
title('误差分析');

% 准确度

pred = predict(Xtrain, THETA, WEIGHT, unit_num_list);
pred_raw = zeros(size(pred,1), 1);  % raw是指标签用1, 2, 3表示, 而不是010, 100这样子
for i = 1:size(pred, 1)
    find(pred(i, :), 1);
    pred_raw(i, :) = find(pred(i, :), 1);
end

fprintf('训练集准确度: %f\n\n', mean(double(pred_raw == ytrain_raw)) * 100);


%% 预测

fprintf('---- 第 3 阶段: 预测 ----\n\n');

fprintf('<按任意键开始预测>\n')

pause;

pred = predict(Xtest, THETA, WEIGHT, unit_num_list);
pred_raw = zeros(size(pred,1), 1);  % raw是指标签用1, 2, 3表示, 而不是010, 100这样子
for i = 1:size(pred, 1)
    find(pred(i, :), 1);
    pred_raw(i, :) = find(pred(i, :), 1);
end

fprintf('测试集准确度: %f\n\n', mean(double(pred_raw == ytest_raw)) * 100);

























