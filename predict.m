%% 预测
function pred = predict(X, THETA, WEIGHT, unit_num_list)

% 定义OUTPUT

OUTPUT = cell(size(unit_num_list, 2), 1);

for i = 2:size(OUTPUT, 1)
    OUTPUT{i} = zeros(unit_num_list(i), 1);
end

pred = [];

% 前向传播
for i = 1:size(X, 1)     % 第i个样本
    OUTPUT{1} = X(i, :)';
    for j = 1:size(OUTPUT, 1) - 1     % 第k层       
        OUTPUT{j + 1} = sigmoid(WEIGHT{j}' * OUTPUT{j} - THETA{j});
    end
    
    [val, index] = max(OUTPUT{end}');
    temp = zeros(1, size(OUTPUT{end}', 2));
    temp(index) = 1;
    pred = [pred; temp];
end

end