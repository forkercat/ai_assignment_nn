%% 王俊皓 2014101027 神经网络作业

clear;
clc;

%% 输入数据和部分参数

fprintf('---- 第 1 阶段: 载入数据 ----\n\n');

% 读入数据
[X, y, y_raw, labels] = loadData('iris.txt', ' ');

% 定义层数, 如3层, 分别有2,2,3个神经元可定义为[2,2,3]
num_input = size(X, 2);
num_output = size(y, 2);
hidden_num_list = [3, 2]; % 可修改隐藏层的层数及每层个数
unit_num_list = [num_input, hidden_num_list, num_output];

% 学习速率
ALPHA = 0.1;

% 迭代次数
ITERATION = 200;

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

fprintf('神经网络层级结构:(输入层 -> 隐藏层 -> 输出层)\n');
fprintf('%d', unit_num_list(1));
for i = 2:size(unit_num_list, 2)
    fprintf(' -> %d', unit_num_list(i));
end
fprintf('\n\n');

fprintf('迭代次数: %d   学习速率: %.4f\n\n', ITERATION, ALPHA);


%% 训练

fprintf('---- 第 2 阶段: 训练模型 ----\n\n');

fprintf('<按任意键开始训练>\n')

pause;

[THETA WEIGHT ERROR] = trainingNN(Xtrain, ytrain, unit_num_list, ALPHA, ITERATION);

% 误差分析
fprintf('\n误差:\n%.5f (第1次迭代) -> %.5f (第%d次迭代)\n\n', sum(ERROR(1,:) .^ 2), sum(ERROR(end,:) .^ 2), ITERATION);


% 准确度
ERROR_X = 1:ITERATION;
ERROR_Y = sum(ERROR .^ 2, 2);

plot(ERROR_X, ERROR_Y, '-');
xlabel('迭代次数');
ylabel('误差');
title('误差分析');

%% 预测

























