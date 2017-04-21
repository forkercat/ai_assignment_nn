%% 激励函数
% 注意z可以是矩阵
function g = sigmoid(z)

g = 1.0 ./ (1.0 + exp(-z));

end