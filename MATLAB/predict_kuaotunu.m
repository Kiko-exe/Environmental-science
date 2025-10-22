function predict_kuaotunu_RF()
% 随机森林回归（纯MATLAB，无工具箱）
% 数据：ERA5(月) + burned area(月)
% 目标：预测月度烧毁面积；默认使用 log1p 变换稳健建模
% 训练集：<=2023；测试集：>=2024

clc; clear; close all;

%% ------------------ 参数 ------------------
paramsRF.nTrees   = 200;   % 森林棵树
paramsRF.maxDepth = 8;     % 单树最大深度
paramsRF.minLeaf  = 3;     % 叶节点最小样本
paramsRF.mtryFrac = 0.5;   % 每次分裂的候选特征比例（~sqrt替代）
paramsRF.qPerFeat = 12;    % 每个特征评估的阈值个数（分位数采样）
useLog1pTarget    = true;  % 对 y 使用 log1p 变换（强烈建议）

%% ------------------ 读取并合并 ------------------
era5_file = 'Kuaotunu_ERA5_monthly_2016_2025.csv';
ba_file   = 'burned area.csv';
assert(isfile(era5_file), '找不到文件：%s', era5_file);
assert(isfile(ba_file),   '找不到文件：%s', ba_file);

E = readtable(era5_file);
B = readtable(ba_file);

[~, ie] = sort(datenum(E.yyyymm, 'yyyy-mm')); E = E(ie,:);
[~, ib] = sort(datenum(B.date_month, 'yyyy-mm')); B = B(ib,:);

M = innerjoin(E, B, 'LeftKeys', 'yyyymm', 'RightKeys', 'date_month');
assert( ~isempty(M), '内连接为空，请检查日期格式是否都是 yyyy-MM。');

% 必要列
needCols = {'T2m_C_mean','RH_pct_mean','PR_mm_sum','WS_ms_mean','WD_deg_mean', ...
            'Cloud_pct_mean','Td_C_mean','U_ms_mean','V_ms_mean', ...
            'year','month','yyyymm','area_ha'};
for c = needCols, assert(ismember(c{1}, M.Properties.VariableNames), '缺少列: %s', c{1}); end

y_raw = M.area_ha;

%% ------------------ 特征工程 ------------------
Xbase = [M.T2m_C_mean, M.RH_pct_mean, M.PR_mm_sum, M.WS_ms_mean, ...
         M.WD_deg_mean, M.Cloud_pct_mean, M.Td_C_mean, M.U_ms_mean, M.V_ms_mean];
namesBase = {'T2m','RH','PR','WS','WDdeg','Cloud','Td','U','V'};

% 季节项
theta = 2*pi*(M.month/12);
Xseas = [sin(theta), cos(theta)];
namesSeas = {'season_sin','season_cos'};

% 滞后 1~3 月（对关键气候+目标做滞后）
lagVars = [M.T2m_C_mean, M.RH_pct_mean, M.PR_mm_sum, M.WS_ms_mean, M.Cloud_pct_mean];
lagNames0 = {'T2m','RH','PR','WS','Cloud'};
maxLag = 3;
Xlags = []; lagNames = {};
for L = 1:maxLag
    Xlags = [Xlags, [nan(L, size(lagVars,2)); lagVars(1:end-L,:)]];
    lagNames = [lagNames, strcat(lagNames0, sprintf('_lag%d', L))];
end
y_lag1 = [nan(1,1); y_raw(1:end-1)];

% 汇总
X = [Xbase, Xseas, Xlags, y_lag1];
featNames = [namesBase, namesSeas, lagNames, {'y_lag1'}];

% 去 NaN
valid = all(~isnan(X),2) & ~isnan(y_raw);
X = X(valid,:); y_raw = y_raw(valid); M2 = M(valid,:);

% 目标变换（稳健）
if useLog1pTarget
    y = log1p(y_raw);
    invTransform = @(z) exp(z) - 1;
    yLabel = 'Burned Area (ha)';
else
    y = y_raw;
    invTransform = @(z) z;
    yLabel = 'Burned Area (ha)';
end

%% ------------------ 划分训练/测试 ------------------
isTrain = M2.year <= 2023;
isTest  = M2.year >= 2024;

Xtr = X(isTrain,:); ytr = y(isTrain);
Xte = X(isTest,:);  yte = y(isTest);

%% ------------------ 训练随机森林 ------------------
fprintf('训练随机森林：%d trees, maxDepth=%d, minLeaf=%d ...\n', ...
    paramsRF.nTrees, paramsRF.maxDepth, paramsRF.minLeaf);

[forest, impSum] = fitRF(Xtr, ytr, paramsRF);

% 特征重要性（按总方差增益）
impTable = table(featNames(:), impSum(:), 'VariableNames', {'Feature','Importance'});
impTable = sortrows(impTable, 'Importance', 'descend');
disp('Top-15 Feature Importance:'); disp(impTable(1:min(15,height(impTable)),:));

%% ------------------ 预测与评估 ------------------
yhat_tr = predictRF(forest, Xtr);
yhat_te = predictRF(forest, Xte);

% 反变换回原始 ha
yhat_tr_ha = invTransform(yhat_tr);
yhat_te_ha = invTransform(yhat_te);
y_tr_ha    = invTransform(ytr);
y_te_ha    = invTransform(yte);

metrics_tr = metrics(y_tr_ha, yhat_tr_ha);
metrics_te = metrics(y_te_ha, yhat_te_ha);

fprintf('\n=== 训练集 ===\n'); disp(metrics_tr);
fprintf('=== 测试集 ===\n');   disp(metrics_te);

%% ------------------ 作图 ------------------
tt = datetime(M2.yyyymm(isTest),'InputFormat','yyyy-MM');
figure('Color','w','Position',[100 100 1100 520]);
plot(tt, y_te_ha, '-o','LineWidth',1.6); hold on;
plot(tt, yhat_te_ha, '-s','LineWidth',1.6);
grid on; xlabel('Month'); ylabel(yLabel);
title('Random Forest — Actual vs Predicted (Test)');
legend({'Actual','RF'}, 'Location','best');

figure('Color','w','Position',[100 100 950 420]);
scatter(y_te_ha, yhat_te_ha, 55, 'filled'); grid on; lsline;
xlabel('Actual ha'); ylabel('Pred RF ha'); title('Test Scatter: RF');

% 重要性条形图
figure('Color','w','Position',[100 100 900 600]);
bar(impTable.Importance(1:min(20,height(impTable))));
set(gca,'XTick',1:min(20,height(impTable)),'XTickLabel',impTable.Feature(1:min(20,height(impTable))));
xtickangle(45); ylabel('Total Variance Gain'); title('Feature Importance (Top-20)'); grid on;

%% ------------------ 保存 ------------------
out = table(M2.yyyymm(isTest), y_te_ha, yhat_te_ha, ...
    'VariableNames', {'yyyymm','Actual_ha','Pred_RF_ha'});
writetable(out, 'predictions_test_RF.csv');
writetable(impTable, 'feature_importance_RF.csv');
writetable(M2, 'merged_model_data_RF.csv');
fprintf('\n✅ 已保存：predictions_test_RF.csv, feature_importance_RF.csv, merged_model_data_RF.csv\n');

end

%% ================== 工具函数 ==================
function [forest, featImp] = fitRF(X, y, p)
% 训练随机森林：Bagging + 随机子特征
N = size(X,1); D = size(X,2);
mtry = max(1, round(p.mtryFrac * D));
forest = cell(p.nTrees,1);
featImp = zeros(D,1);

for t = 1:p.nTrees
    % 自助采样
    idx = randi(N, N, 1);
    Xb = X(idx,:); yb = y(idx);
    % 训练一棵树
    [tree, imp] = fitTree(Xb, yb, 0, p, mtry);
    forest{t} = tree;
    featImp = featImp + imp; % 累计重要性
end
featImp = featImp / p.nTrees;
end

function [tree, featImp] = fitTree(X, y, depth, p, mtry)
% 递归训练 CART 回归树（随机子特征 + 分位数阈值采样）
node.isLeaf = false; node.pred = mean(y);
node.splitFeat = []; node.splitThresh = [];
node.left = []; node.right = [];
featImp = zeros(size(X,2),1);

N = size(X,1);
if depth >= p.maxDepth || N <= 2*p.minLeaf || var(y) < 1e-12
    node.isLeaf = true; tree = node; return;
end

D = size(X,2); 
featIdx = randperm(D, mtry);        % 候选特征
bestGain = 0; bestF = []; bestThr = []; bestL = []; bestR = [];

parentVar = var(y) * N;             % 用方差*样本数做不纯度
for f = featIdx
    x = X(:,f);
    if numel(unique(x)) <= 5, continue; end
    % 分位数候选阈值
    qs = linspace(0.1, 0.9, p.qPerFeat);
    thrCand = unique(quantile(x, qs));
    for th = thrCand(:)'
        L = x <= th; R = ~L;
        nL = sum(L); nR = sum(R);
        if nL < p.minLeaf || nR < p.minLeaf, continue; end
        varL = var(y(L)) * nL; varR = var(y(R)) * nR;
        gain = parentVar - (varL + varR);
        if gain > bestGain
            bestGain = gain; bestF = f; bestThr = th; bestL = L; bestR = R;
        end
    end
end

if isempty(bestF) % 无有效切分 → 叶子
    node.isLeaf = true; tree = node; return;
end

% 记录重要性（方差减少）
featImp(bestF) = featImp(bestF) + bestGain;

% 递归左右子树
[nodeL, impL] = fitTree(X(bestL,:), y(bestL), depth+1, p, mtry);
[nodeR, impR] = fitTree(X(bestR,:), y(bestR), depth+1, p, mtry);
featImp = featImp + impL + impR;

node.splitFeat = bestF;
node.splitThresh = bestThr;
node.left = nodeL; node.right = nodeR;

tree = node;
end

function yhat = predictRF(forest, X)
T = numel(forest);
N = size(X,1);
pred = zeros(N,T);
for t = 1:T
    pred(:,t) = predictTree(forest{t}, X);
end
yhat = mean(pred, 2);
end

function yhat = predictTree(tree, X)
N = size(X,1); yhat = zeros(N,1);
for i = 1:N
    node = tree;
    while ~node.isLeaf
        if X(i, node.splitFeat) <= node.splitThresh
            node = node.left;
        else
            node = node.right;
        end
    end
    yhat(i) = node.pred;
end
end

function M = metrics(y, yhat)
y = y(:); yhat = yhat(:);
rmse = sqrt(mean((y - yhat).^2));
mae  = mean(abs(y - yhat));
ss_res = sum((y - yhat).^2);
ss_tot = sum((y - mean(y)).^2);
R2 = 1 - ss_res / max(ss_tot, eps);
M = table(rmse, mae, R2, 'VariableNames', {'RMSE','MAE','R2'});
end
