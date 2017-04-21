%% 读取文本
% splitChar是每行的分割字符
% X是特征集, y是标签集, y_unit是适应神经网络输出的格式, labels是标签种类
% 如样本1 2 3 yes
%      2 3 4 no
%      1 3 5 yes
% y = [1 0; 0 1; 1 0]
% y_raw = [1; 2; 1] (注意是从1开始)  
% labels = {'yes', 'no'}
function [X, y, y_raw, labels] = loadData(fileName, splitChar)

fid = fopen(fileName);

X = [];
y = [];
y_raw = [];
labels = {};

while ~feof(fid)
    % X
    line=fgetl(fid);
    result = regexp(line, splitChar, 'split');
    temp = [];
    for i=1:size(result, 2) - 1 % 去掉标签栏
        temp = [temp str2num(cell2mat(result(i)))];
    end
    X = [X; temp];
    
    % label
    labelStr = result(end);
    if sum(ismember(labels, labelStr)) == 0
        % 不在集合中
        labels = [labels labelStr];
    end
    
    y_raw = [y_raw; find(ismember(labels, labelStr), 1)];
    
end;


for i=1:size(y_raw, 1)
    y = [y; ismember(labels, labels{y_raw(i)})];
end


% displaying

fprintf('样本量: %d 个\n特征量: %d 个\n类别: ', size(X, 1), size(X, 2));

for i=1:size(labels, 2)
    fprintf('%s', labels{i});
    if i ~= size(labels, 2)
        fprintf(', ');
    end
end
fprintf('\n\n');

end