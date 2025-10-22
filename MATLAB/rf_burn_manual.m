function burn_predictor_perfect_4f(mode)
% Minimal, robust, toolbox-free pipeline that "just works".
% Features (train & manual input): Temperature, Relative Humidity, Wind Direction, Wind Speed
% Engineering: sin/cos(WD) for circularity + VPD(T,RH)
% Model: Hand-written Random Forest Regressor on log1p(area)
%
% Files required:
%   - Kuaotunu_ERA5_monthly_2016_2025.csv  (must contain yyyymm, year, month, T2m_C_mean, RH_pct_mean, WD_deg_mean, WS_ms_mean)
%   - burned area.csv                      (must contain date_month, area_ha)
%
% Usage:
%   burn_predictor_perfect_4f('train')        % train, evaluate, plots & CSVs
%   burn_predictor_perfect_4f('interactive')  % manual inputs -> prediction (+ interval)

if nargin==0, mode = 'train'; end
switch lower(mode)
    case 'train',        train_and_evaluate();
    case 'interactive',  interactive_predict();
    otherwise, error('Unknown mode. Use ''train'' or ''interactive''.');
end
end

%% ===================== TRAIN + EVALUATE =====================
function train_and_evaluate()
clc; close all;

era5 = 'Kuaotunu_ERA5_monthly_2016_2025.csv';
ba   = 'burned area.csv';
assert(isfile(era5) && isfile(ba), 'CSV files not found.');

E = readtable(era5);
B = readtable(ba);
[~, ie] = sort(datenum(E.yyyymm,'yyyy-mm')); E = E(ie,:);
[~, ib] = sort(datenum(B.date_month,'yyyy-mm')); B = B(ib,:);
M = innerjoin(E, B, 'LeftKeys','yyyymm', 'RightKeys','date_month');

needE = {'yyyymm','year','month','T2m_C_mean','RH_pct_mean','WD_deg_mean','WS_ms_mean'};
needB = {'date_month','area_ha'};
for i=1:numel(needE), assert(ismember(needE{i}, E.Properties.VariableNames), 'Missing: %s', needE{i}); end
for i=1:numel(needB), assert(ismember(needB{i}, B.Properties.VariableNames), 'Missing: %s', needB{i}); end

% ==== Build features only from the 4 manual variables ====
T  = M.T2m_C_mean;              % °C
RH = M.RH_pct_mean;             % %
WD = M.WD_deg_mean;             % deg
WS = M.WS_ms_mean;              % m/s

[sinWD, cosWD] = deal(sind(WD), cosd(WD));
VPD = compute_vpd(T, RH);       % kPa; key dryness indicator

% Optional simple interactions (still from 4 inputs)
X = [T, RH, WS, sinWD, cosWD, VPD, T.*WS, RH.*WS, T.*RH];
featNames = {'T','RH','WS','sinWD','cosWD','VPD','T_x_WS','RH_x_WS','T_x_RH'};

y_raw = M.area_ha;
ok = all(~isnan(X),2) & ~isnan(y_raw);
X = X(ok,:); y_raw = y_raw(ok); M2 = M(ok,:);

y = log1p(y_raw);               % robust target

% ==== Split by time ====
isTrain = M2.year <= 2023;
isTest  = M2.year >= 2024;

Xtr = X(isTrain,:); ytr = y(isTrain);
Xte = X(isTest,:);  yte = y(isTest);

% ==== Train Random Forest (toolbox-free) ====
p.nTrees   = 500;
p.maxDepth = 12;
p.minLeaf  = 3;
p.mtryFrac = 0.7;
p.qPerFeat = 25;    % number of threshold quantiles per feature

fprintf('Training RF (log1p target): %d trees, depth=%d, minLeaf=%d ...\n', p.nTrees, p.maxDepth, p.minLeaf);
[forest, imp] = fitRF_reg(Xtr, ytr, p);

% ==== Predict ====
yhat_tr_each = predictRF_all(forest, Xtr);   % per-tree predictions
yhat_te_each = predictRF_all(forest, Xte);

yhat_tr = mean(yhat_tr_each,2);
yhat_te = mean(yhat_te_each,2);

% Back-transform to ha
ytr_ha    = exp(ytr) - 1;
yte_ha    = exp(yte) - 1;
yhat_tr_ha= max(0, exp(yhat_tr) - 1);
yhat_te_ha= max(0, exp(yhat_te) - 1);

% Simple ~95% intervals (tree spread in log-space)
lo_tr = exp(qtile(yhat_tr_each, 0.025, 2)) - 1; lo_tr = max(0, lo_tr);
hi_tr = exp(qtile(yhat_tr_each, 0.975, 2)) - 1;
lo_te = exp(qtile(yhat_te_each, 0.025, 2)) - 1; lo_te = max(0, lo_te);
hi_te = exp(qtile(yhat_te_each, 0.975, 2)) - 1;

% ==== Metrics ====
met_tr = metrics(ytr_ha, yhat_tr_ha);
met_te = metrics(yte_ha, yhat_te_ha);
fprintf('\n=== TRAIN ===\n'); disp(met_tr);
fprintf('=== TEST  ===\n');  disp(met_te);

% Extra: MAPE on non-zero actuals
mask = yte_ha > 0;
mape = NaN;
if any(mask), mape = mean(abs((yte_ha(mask)-yhat_te_ha(mask))./yte_ha(mask)))*100; end

% ==== Save model ====
model.forest     = forest;
model.featNames  = featNames;
model.params     = p;
model.transform  = 'log1p';
save('rf_model_perfect_4f.mat','model');
fprintf('✅ Saved model: rf_model_perfect_4f.mat\n');

% ==== Plots ====
tt = datetime(M2.yyyymm(isTest),'InputFormat','yyyy-MM');

figure('Color','w','Position',[80 80 1100 520]);
plot(tt, yte_ha,      '-o','LineWidth',1.4); hold on;
plot(tt, yhat_te_ha,  '-s','LineWidth',1.4);
plot(tt, lo_te, '--', 'LineWidth',1.0);
plot(tt, hi_te, '--', 'LineWidth',1.0);
grid on; xlabel('Month'); ylabel('Burned Area (ha)');
title('RF (4 inputs) — Actual vs Predicted (Test)'); 
legend({'Actual','Pred','Pred lo','Pred hi'},'Location','best');
saveas(gcf, 'plot_test_timeseries_RF_4f.png');

figure('Color','w','Position',[80 80 900 400]);
scatter(yte_ha, yhat_te_ha, 55, 'filled'); grid on;
xlabel('Actual (ha)'); ylabel('Predicted (ha)'); title('RF (4 inputs) — Test Scatter');
saveas(gcf, 'plot_test_scatter_RF_4f.png');

res = yte_ha - yhat_te_ha;
figure('Color','w','Position',[80 80 900 400]);
histogram(res, 15); grid on;
xlabel('Residual (ha)'); ylabel('Count'); title('Residuals Histogram (Test)');
saveas(gcf, 'plot_residuals_hist_RF_4f.png');

figure('Color','w','Position',[80 80 900 400]);
scatter(yhat_te_ha, res, 35, 'filled'); yline(0,'--'); grid on;
xlabel('Fitted (ha)'); ylabel('Residual (ha)'); title('Residuals vs Fitted (Test)');
saveas(gcf, 'plot_resid_vs_fitted_RF_4f.png');

% ==== CSV exports ====
pred_tbl = table(M2.yyyymm(isTest), yte_ha, yhat_te_ha, lo_te, hi_te, ...
   'VariableNames',{'yyyymm','Actual_ha','Pred_ha','Pred_lo_ha','Pred_hi_ha'});
writetable(pred_tbl,'predictions_test_RF_4f.csv');

impTbl = table(featNames(:), imp(:), 'VariableNames', {'Feature','Importance'});
impTbl = sortrows(impTbl,'Importance','descend');
writetable(impTbl,'feature_importance_RF_4f.csv');

metTbl = table(met_tr.rmse,met_tr.mae,met_tr.R2, met_te.rmse,met_te.mae,met_te.R2, mape, ...
    'VariableNames',{'RMSE_train','MAE_train','R2_train','RMSE_test','MAE_test','R2_test','MAPE_test_pct'});
writetable(metTbl,'metrics_RF_4f.csv');

merge_out = M2(:, {'yyyymm','year','month','T2m_C_mean','RH_pct_mean','WD_deg_mean','WS_ms_mean','area_ha'});
writetable(merge_out,'merged_data_RF_4f.csv');

fprintf('✅ Saved: predictions_test_RF_4f.csv, feature_importance_RF_4f.csv, metrics_RF_4f.csv, merged_data_RF_4f.csv\n');
end

%% ===================== INTERACTIVE PREDICTION =====================
function interactive_predict()
if ~isfile('rf_model_perfect_4f.mat')
    fprintf('No model found. Training now...\n');
    train_and_evaluate();
end
S = load('rf_model_perfect_4f.mat'); model = S.model;

fprintf('\n=== Interactive Prediction (English prompts; 4 inputs) ===\n');
keep = true;
while keep
    T  = asknum('Temperature (°C)', 22);
    RH = asknum('Relative Humidity (%)', 40);
    WD = input('Wind Direction (cardinal N/NE/E/SE/S/SW/W/NW OR 0–360 deg): ', 's');
    WS = asknum('Wind Speed (m/s)', 6);

    WDdeg = parse_wind_dir(WD);
    [sinWD, cosWD] = deal(sind(WDdeg), cosd(WDdeg));
    VPD = compute_vpd(T, RH);

    x = [T, RH, WS, sinWD, cosWD, VPD, T*WS, RH*WS, T*RH];
    y_each = predictRF_all(model.forest, x);
    y_mean = mean(y_each,2);
    y_lo   = qtile(y_each,0.025,2);
    y_hi   = qtile(y_each,0.975,2);

    pred = max(0, exp(y_mean) - 1);
    lo   = max(0, exp(y_lo)   - 1);
    hi   = max(0, exp(y_hi)   - 1);

    fprintf('\n>>> Predicted Burned Area: %.2f ha', pred);
    fprintf('   (~95%% interval: %.2f – %.2f ha)\n\n', lo, hi);

    s = input('Another prediction? (y/n) ','s');
    if isempty(s) || lower(s(1))~='y', keep = false; end
end
fprintf('Done.\n');
end

%% ===================== RF (toolbox-free) =====================
function [forest, featImp] = fitRF_reg(X, y, p)
N = size(X,1); D = size(X,2);
mtry = max(1, round(p.mtryFrac * D));
forest = cell(p.nTrees,1);
featImp = zeros(D,1);
for t = 1:p.nTrees
    idx = randi(N,N,1); Xb = X(idx,:); yb = y(idx);
    [tree, imp] = fitTreeR(Xb, yb, 0, p, mtry);
    forest{t} = tree;
    featImp = featImp + imp;
end
featImp = featImp / p.nTrees;
end

function [tree, featImp] = fitTreeR(X, y, depth, p, mtry)
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
    qs = linspace(0.1, 0.9, p.qPerFeat);
    thrCand = unique(qtile(x, qs, 1));   % our own quantile (no toolbox)
    for th = thrCand(:)'
        L = x<=th; R = ~L; nL=sum(L); nR=sum(R);
        if nL < p.minLeaf || nR < p.minLeaf, continue; end
        varL = var(y(L)) * nL; varR = var(y(R)) * nR;
        gain = parentVar - (varL + varR);
        if gain > bestGain
            bestGain=gain; bestF=f; bestThr=th; bestL=L; bestR=R;
        end
    end
end

if isempty(bestF)
    node.isLeaf=true; tree=node; return;
end

featImp(bestF) = featImp(bestF) + bestGain;
[nodeL, impL] = fitTreeR(X(bestL,:), y(bestL), depth+1, p, mtry);
[nodeR, impR] = fitTreeR(X(bestR,:), y(bestR), depth+1, p, mtry);
featImp = featImp + impL + impR;

node.splitFeat=bestF; node.splitThresh=bestThr;
node.left=nodeL; node.right=nodeR;
tree=node;
end

function P = predictRF_all(forest, X)
if isrow(X), X = X(:)'; end
N = size(X,1); T = numel(forest);
P = zeros(N,T);
for t=1:T, P(:,t) = predictTreeR(forest{t}, X); end
end

function yhat = predictTreeR(tree, X)
N=size(X,1); yhat=zeros(N,1);
for i=1:N
    node=tree;
    while ~node.isLeaf
        if X(i,node.splitFeat) <= node.splitThresh
            node=node.left;
        else
            node=node.right;
        end
    end
    yhat(i)=node.pred;
end
end

%% ===================== HELPERS =====================
function VPD = compute_vpd(T, RH)
% Tetens formula; T in °C; RH in %
es = 0.6108 .* exp((17.27.*T) ./ (T + 237.3));
ea = es .* (RH/100);
VPD = max(0, es - ea);
end

function v = asknum(prompt, defaultV)
s = input(sprintf('%s [default %.2f]: ', prompt, defaultV), 's');
if isempty(s), v = defaultV; else, v = str2double(s); end
if isnan(v), v = defaultV; end
end

function deg = parse_wind_dir(inp)
% Accepts cardinal (N/NE/E/SE/S/SW/W/NW) or numeric degrees
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
        d = str2double(card); if isnan(d), d=0; end
        deg = mod(d,360);
end
end

function q = qtile(x, qv, dim)
% Toolbox-free quantile. qv in [0,1], scalar or vector.
% If dim omitted for matrices from RF (N x T), use dim=2 (row-wise).
if nargin<3, dim = 1; end
if isvector(x)
    xs = sort(x(:));
    n = numel(xs);
    q = zeros(size(qv));
    for k=1:numel(qv)
        qk = min(max(qv(k),0),1);
        if n==1, q(k)=xs(1); continue; end
        pos = 1 + (n-1)*qk;
        i = floor(pos); j = ceil(pos);
        if i==j, q(k)=xs(i); else
            w = pos - i; q(k) = (1-w)*xs(i) + w*xs(j);
        end
    end
else
    % operate along rows (dim=2) typically
    sz = size(x);
    if dim==2
        q = zeros(sz(1), numel(qv));
        for r=1:sz(1)
            q(r,:) = qtile(x(r,:), qv, 1);
        end
    else
        q = zeros(numel(qv), sz(2));
        for c=1:sz(2)
            q(:,c) = qtile(x(:,c), qv, 1);
        end
    end
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
