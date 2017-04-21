%% 随机初始化参数函数, size是返回的尺寸, F是输入层的单元个数
function R = random_initialize(size, F)

R = rand(size(1), size(2)); % 初始化0 - 1
R = R .* 2 - 1;             % 范围变换到-1 - 1
R = (2.4 / F) .* R;         % Haykin的公式

end