%% 归一化
function [X_norm, maxVal, minVal] = featureNormalize(X)

X_norm = X;

maxVal = max(X);
minVal = min(X);

% 样本个数
m = size(X_norm, 1);

% 生成多列一样的max和min
maxVal = repmat(maxVal, m, 1);
minVal = repmat(minVal, m, 1);


X_norm = (X_norm - minVal) ./ (maxVal - minVal);


end
