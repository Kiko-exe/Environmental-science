function burn_predictor_cli_en()
% Interactive wildfire area predictor (English prompts)
% - First run auto-trains a Random Forest (pure MATLAB, no toolboxes)
% - Inputs (in English): Temperature, Relative Humidity, Wind Direction, Wind Speed, Precipitation
% - Outputs predicted burned area (ha) + ~95% interval
% - Trains on your CSVs and saves error analysis (metrics + residual plots)

clc; close all;
modelFile = 'rf_model_manual_en.mat';
if ~isfile(modelFile)
    fprintf('No model found. Training a model first...\n');
    train_model_manual_inputs_en(modelFile);
else
    fprintf('Found model: %s\n', modelFile);
end

S = load(modelFile); model = S.model;

fprintf('\n=== Interactive Prediction (English prompts) ===\n');
keep = true;
while keep
    T  = asknum_en('Temperature (°C)', 22);
    RH = asknum_en('Relative Humidity (%)', 40);
    WD = input('Wind Direction (cardinal N/NE/E/SE/S/SW/W/NW OR 0–360 degrees): ','s');
    WS = asknum_en('Wind Speed (m/s)', 6);
    PR = asknum_en('Precipitation (mm)', 50);

    WDdeg = parse_wind_dir_en(WD);
    x = [T, RH, WDdeg, WS, PR];

    [yhat_ha, lo_ha, hi_ha] = predict_burn_from_inputs_en(model, x);

    fprintf('\n>>> Predicted Burned Area: %.2f ha', yhat_ha);
    fprintf('    (~95%% interval: %.2f – %.2f ha)\n\n', lo_ha, hi_ha);

    s = input('Do you want another prediction? (y/n) ','s');
    if isempty(s) || lower(s(1))~='y', keep=false; end
end
fprintf('Done.\n');
end

%% ================= TRAIN (5 features: T, RH, WD, WS, PR) =================
function train_model_manual_inputs_en(modelFile)
era5_file = 'Kuaotunu_ERA5_monthly_2016_2025.csv';
ba_file   = 'burned area.csv';
assert(isfile(era5_file) && isfile(ba_file), 'CSV files not found.');

E = readtable(era5_file);
B = readtable(ba_file);
[~, ie] = sort(datenum(E.yyyymm,'yyyy-mm')); E = E(ie,:);
[~, ib] = sort(datenum(B.date_month,'yyyy-mm')); B = B(ib,:);

M = innerjoin(E, B, 'LeftKeys','yyyymm', 'RightKeys','date_month');
need = {'T2m_C_mean','RH_pct_mean','WD_deg_mean','WS_ms_mean','PR_mm_sum','year','yyyymm','area_ha'};
for i=1:numel(need), assert(ismember(need{i},M.Properties.VariableNames), 'Missing column: %s', need{i}); end

% Features: temperature, humidity, wind direction (deg), wind speed, precipitation
X = [M.T2m_C_mean, M.RH_pct_mean, M.WD_deg_mean, M.WS_ms_mean, M.PR_mm_sum];
y_raw = M.area_ha;

ok = all(~isnan(X),2) & ~isnan(y_raw);
X = X(ok,:); y_raw = y_raw(ok); M = M(ok,:);

% Robust target transform
y = log1p(y_raw);

% Train/test split (by time)
isTrain = M.year <= 2023; isTest = M.year >= 2024;
Xtr = X(isTrain,:); ytr = y(isTrain);
Xte = X(isTest,:);  yte = y(isTest);

% RF params
p.nTrees   = 400;
p.maxDepth = 12;
p.minLeaf  = 3;
p.mtryFrac = 0.8;
p.qPerFeat = 20;

fprintf('Training Random Forest (5 features): %d trees, maxDepth=%d, minLeaf=%d ...\n', ...
    p.nTrees, p.maxDepth, p.minLeaf);
[forest, imp] = fitRF_en(Xtr, ytr, p);

% Evaluate (inverse-transform to ha)
yhat_tr = predictRF_mean_en(forest, Xtr);
yhat_te = predictRF_mean_en(forest, Xte);
met_tr  = metrics_en(expm1(ytr),  expm1(yhat_tr));
met_te  = metrics_en(expm1(yte),  expm1(yhat_te));
fprintf('\n=== TRAIN ===\n'); disp(met_tr);
fprintf('=== TEST  ===\n');  disp(met_te);

% Save model
model.forest    = forest;
model.varsOrder = {'T2m_C_mean','RH_pct_mean','WD_deg_mean','WS_ms_mean','PR_mm_sum'};
model.transform = 'log1p';
model.params    = p;
save(modelFile, 'model');
fprintf('✅ Model saved: %s\n', modelFile);

% Feature importance
featNames = model.varsOrder(:);
impTable = table(featNames, imp(:), 'VariableNames', {'Feature','Importance'});
impTable = sortrows(impTable,'Importance','descend');
disp('Feature Importance:'); disp(impTable);

% ---------- Error analysis outputs ----------
% 1) Save test predictions CSV
pred_tbl = table(M.yyyymm(isTest), expm1(yte), expm1(yhat_te), ...
    'VariableNames', {'yyyymm','Actual_ha','Pred_RF_ha'});
writetable(pred_tbl, 'predictions_test_RF_EN.csv');
fprintf('✅ Saved: predictions_test_RF_EN.csv\n');

% 2) Residuals on test
res_te = expm1(yte) - expm1(yhat_te);
res_tbl = table(M.yyyymm(isTest), res_te, 'VariableNames', {'yyyymm','Residual_ha'});
writetable(res_tbl, 'residuals_test_RF_EN.csv');
fprintf('✅ Saved: residuals_test_RF_EN.csv\n');

% 3) Plots: time series + scatter + residual histogram + residual vs fitted
tt = datetime(M.yyyymm(isTest),'InputFormat','yyyy-MM');

figure('Color','w','Position',[80 80 1100 520]);
plot(tt, expm1(yte), '-o','LineWidth',1.4); hold on;
plot(tt, expm1(yhat_te), '-s','LineWidth',1.4);
grid on; xlabel('Month'); ylabel('Burned Area (ha)');
title('RF — Actual vs Predicted (Test)'); legend({'Actual','RF'},'Location','best');
saveas(gcf, 'plot_test_timeseries_RF_EN.png');

figure('Color','w','Position',[80 80 900 400]);
scatter(expm1(yte), expm1(yhat_te), 55, 'filled'); grid on; lsline;
xlabel('Actual (ha)'); ylabel('Predicted (ha)'); title('RF — Test Scatter');
saveas(gcf, 'plot_test_scatter_RF_EN.png');

figure('Color','w','Position',[80 80 900 400]);
histogram(res_te, 15); grid on;
xlabel('Residual (ha)'); ylabel('Count'); title('RF — Residuals Histogram (Test)');
saveas(gcf, 'plot_residuals_hist_RF_EN.png');

figure('Color','w','Position',[80 80 900 400]);
scatter(expm1(yhat_te), res_te, 35, 'filled'); grid on; yline(0,'--');
xlabel('Fitted (ha)'); ylabel('Residual (ha)'); title('RF — Residuals vs Fitted (Test)');
saveas(gcf, 'plot_resid_vs_fitted_RF_EN.png');

% 4) Extra error metrics: MAPE (on non-zero actuals)
mask = expm1(yte) > 0;
if any(mask)
    mape = mean(abs((expm1(yte(mask)) - expm1(yhat_te(mask))) ./ expm1(yte(mask)))) * 100;
else
    mape = NaN;
end
extra_metrics = table(mape, 'VariableNames', {'MAPE_percent'});
writetable(extra_metrics, 'extra_metrics_RF_EN.csv');
fprintf('✅ Saved: extra_metrics_RF_EN.csv (MAPE on non-zero months)\n');

% 5) Save importance
writetable(impTable, 'feature_importance_RF_EN.csv');
fprintf('✅ Saved: feature_importance_RF_EN.csv\n');
end

%% ================ PREDICT (mean & ~95% interval) =================
function [yhat_ha, lo_ha, hi_ha] = predict_burn_from_inputs_en(model, xrow)
if isrow(xrow), xrow = xrow(:)'; end
yhat_each = predictRF_all_en(model.forest, xrow);
yhat_mean = mean(yhat_each, 2);
yhat_std  = std(yhat_each, 0, 2);

if isfield(model,'transform') && strcmp(model.transform,'log1p')
    yhat_ha = max(0, exp(yhat_mean) - 1);
    lo_ha   = max(0, exp(yhat_mean - 1.96*yhat_std) - 1);
    hi_ha   = max(0, exp(yhat_mean + 1.96*yhat_std) - 1);
else
    yhat_ha = yhat_mean;
    lo_ha   = yhat_mean - 1.96*yhat_std;
    hi_ha   = yhat_mean + 1.96*yhat_std;
    lo_ha   = max(0, lo_ha);
end
end

%% ==================== INPUT HELPERS (EN) ====================
function v = asknum_en(prompt, defaultV)
s = input(sprintf('%s [default %.2f]: ', prompt, defaultV), 's');
if isempty(s), v = defaultV; else, v = str2double(s); end
if isnan(v), v = defaultV; end
end

function deg = parse_wind_dir_en(inp)
% Accepts cardinal or numeric degrees
if isempty(inp), deg = 0; return; end
if isnumeric(inp), deg = mod(double(inp),360); return; end
card = upper(strtrim(string(inp)));
switch char(card)
    case 'N',  deg = 0;
    case 'NNE',deg = 22.5;
    case 'NE', deg = 45;
    case 'ENE',deg = 67.5;
    case 'E',  deg = 90;
    case 'ESE',deg = 112.5;
    case 'SE', deg = 135;
    case 'SSE',deg = 157.5;
    case 'S',  deg = 180;
    case 'SSW',deg = 202.5;
    case 'SW', deg = 225;
    case 'WSW',deg = 247.5;
    case 'W',  deg = 270;
    case 'WNW',deg = 292.5;
    case 'NW', deg = 315;
    case 'NNW',deg = 337.5;
    otherwise
        d = str2double(card);
        if isnan(d), d = 0; end
        deg = mod(d,360);
end
end

%% ==================== RANDOM FOREST (pure MATLAB) ====================
function [forest, featImp] = fitRF_en(X, y, p)
N = size(X,1); D = size(X,2);
mtry = max(1, round(p.mtryFrac * D));
forest = cell(p.nTrees,1);
featImp = zeros(D,1);
for t = 1:p.nTrees
    idx = randi(N,N,1); Xb = X(idx,:); yb = y(idx);
    [tree, imp] = fitTree_en(Xb, yb, 0, p, mtry);
    forest{t} = tree; featImp = featImp + imp;
end
featImp = featImp / p.nTrees;
end

function [tree, featImp] = fitTree_en(X, y, depth, p, mtry)
node.isLeaf=false; node.pred=mean(y);
node.splitFeat=[]; node.splitThresh=[]; node.left=[]; node.right=[];
featImp = zeros(size(X,2),1);
N = size(X,1);
if depth>=p.maxDepth || N<=2*p.minLeaf || var(y)<1e-12
    node.isLeaf=true; tree=node; return;
end
D = size(X,2);
featIdx = randperm(D, mtry);
bestGain=0; bestF=[]; bestThr=[]; bestL=[]; bestR=[];
parentVar = var(y) * N;
for f = featIdx
    x = X(:,f);
    if numel(unique(x)) <= 5, continue; end
    qs = linspace(0.1,0.9,p.qPerFeat);
    thrCand = unique(quantile(x, qs));
    for th = thrCand(:)'
        L = x<=th; R = ~L; nL=sum(L); nR=sum(R);
        if nL<p.minLeaf || nR<p.minLeaf, continue; end
        gain = parentVar - (var(y(L))*nL + var(y(R))*nR);
        if gain>bestGain, bestGain=gain; bestF=f; bestThr=th; bestL=L; bestR=R; end
    end
end
if isempty(bestF), node.isLeaf=true; tree=node; return; end
featImp(bestF) = featImp(bestF) + bestGain;
[nodeL, impL] = fitTree_en(X(bestL,:), y(bestL), depth+1, p, mtry);
[nodeR, impR] = fitTree_en(X(bestR,:), y(bestR), depth+1, p, mtry);
featImp = featImp + impL + impR;
node.splitFeat=bestF; node.splitThresh=bestThr; node.left=nodeL; node.right=nodeR;
tree = node;
end

function yhat = predictRF_mean_en(forest, X)
if isrow(X), X = X(:)'; end
N = size(X,1); T = numel(forest);
P = zeros(N,T);
for t=1:T, P(:,t) = predictTree_en(forest{t}, X); end
yhat = mean(P,2);
end

function P = predictRF_all_en(forest, X)
if isrow(X), X = X(:)'; end
N = size(X,1); T = numel(forest);
P = zeros(N,T);
for t=1:T, P(:,t) = predictTree_en(forest{t}, X); end
end

function yhat = predictTree_en(tree, X)
N = size(X,1); yhat = zeros(N,1);
for i=1:N
    node = tree;
    while ~node.isLeaf
        if X(i,node.splitFeat) <= node.splitThresh
            node = node.left;
        else
            node = node.right;
        end
    end
    yhat(i) = node.pred;
end
end

function M = metrics_en(y, yhat)
y = y(:); yhat = yhat(:);
rmse = sqrt(mean((y - yhat).^2));
mae  = mean(abs(y - yhat));
ss_res = sum((y - yhat).^2);
ss_tot = sum((y - mean(y)).^2);
R2 = 1 - ss_res / max(ss_tot, eps);
M = table(rmse, mae, R2, 'VariableNames', {'RMSE','MAE','R2'});
end

bar(importance)
set(gca,'xticklabels',{'Temp','RH','Precip','WindSpeed'})
ylabel('Predictor Importance')
title('Variable Importance (RF Model)')
