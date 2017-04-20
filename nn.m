%% 输入参数
clear
% 定义层数, 如3层, 分别有2,2,3个神经元可定义为[2,2,3]

num_input = 2;
num_output = 2;

unit_num_list = [num_input, 3, 2, num_output];

%% 初始化权值和阈值

% 层数 (减去输入层和输出层)
num_hidden_layer = size(unit_num_list, 2) - 2;

% 定义权重
WEIGHT = cell(num_hidden_layer + 1, 1);

for i = 1:num_hidden_layer + 1
    WEIGHT{i} = rand(unit_num_list(i), unit_num_list(i + 1));
    WEIGHT{i} = WEIGHT{i} .* 2 - 1; % 控制在-1,1之间
    WEIGHT{i} = random_initialize(WEIGHT{i}, num_input);
end

num_WEIGHT = num_hidden_layer + 1;

% 定义阈值

THETA = cell(num_hidden_layer, 1);

for i = 1:num_hidden_layer
    THETA{i} = rand(unit_num_list(i + 1), 1);
    THETA{i} = THETA{i} .* 2 - 1;
    THETA{i} = random_initialize(THETA{i}, num_input);
end

%% 

    


