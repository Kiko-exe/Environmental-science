%% ==================== Basic Fire Model (No Toolbox, Fully Compatible) ====================
% Works on any MATLAB installation, no toolboxes required.
% Uses manual ensemble (bagging) + basic math.
%
% Input files in same folder:
%   1) burned area.csv  -> date_month, area_ha
%   2) Kuaotunu_ERA5_monthly_2016_2025.csv  -> climate variables (T2m_C_mean etc.)

close all; clear; clc;

%% ---------------- 1. Read and merge data ----------------
burn = readtable('burned area.csv');
cli  = readtable('Kuaotunu_ERA5_monthly_2016_2025.csv');

% Parse date
burn.date = datetime(burn.date_month,'InputFormat','yyyy-MM','Format','yyyy-MM');
burn.year = year(burn.date);
burn.month = month(burn.date);

% Merge
T = innerjoin(cli, burn(:,{'year','month','area_ha'}), 'Keys',{'year','month'});
T = sortrows(T, {'year','month'});
T.dateYM = datetime(T.year, T.month, 1);

%% ---------------- 2. Feature engineering ----------------
% Calculate VPD (kPa)
T.es_kPa = 0.6108 .* exp(17.27 .* T.T2m_C_mean ./ (T.T2m_C_mean + 237.3));
T.ea_kPa = T.es_kPa .* (T.RH_pct_mean ./ 100);
T.VPD_kPa = T.es_kPa - T.ea_kPa;

% Wind direction to sin/cos if exists
if ismember('WD_deg_mean', T.Properties.VariableNames)
    T.WindDir_sin = sind(T.WD_deg_mean);
    T.WindDir_cos = cosd(T.WD_deg_mean);
    hasDir = true;
else
    hasDir = false;
end

% Build predictor matrix
predictors = {'T2m_C_mean','RH_pct_mean','PR_mm_sum','WS_ms_mean','VPD_kPa'};
if hasDir
    predictors = [predictors, {'WindDir_sin','WindDir_cos'}];
end
X = T{:, predictors};
y = T.area_ha;

% Remove NaNs
good = all(~isnan(X),2) & ~isnan(y);
X = X(good,:); y = y(good); T = T(good,:);

%% ---------------- 3. Split data (chronological 70/30) ----------------
n = size(X,1);
nTrain = floor(0.7 * n);
Xtrain = X(1:nTrain,:); ytrain = y(1:nTrain);
Xtest  = X(nTrain+1:end,:); ytest = y(nTrain+1:end);
dateTest = T.dateYM(nTrain+1:end);

%% ---------------- 4. Manual Ensemble (bagged mean model) ----------------
numModels = 100;
nObs = size(Xtrain,1);
rng(42)
yhat_train_all = zeros(nTrain,numModels);
yhat_test_all  = zeros(length(ytest),numModels);

fprintf('Training %d simple ensemble models (no toolbox)...\n', numModels);
for i = 1:numModels
    % Bootstrap sample indices
    idx = randi(nObs, nObs, 1);
    % Linear relationship per variable (manual regression)
    w = zeros(size(Xtrain,2),1);
    for j = 1:size(Xtrain,2)
        xj = Xtrain(idx,j);
        w(j) = (xj' * ytrain(idx)) / (xj' * xj + 1e-6);  % simple slope
    end
    % Predict
    yhat_train_all(:,i) = Xtrain * w;
    yhat_test_all(:,i)  = Xtest  * w;
end

% Ensemble average
yhat_train = mean(yhat_train_all,2);
yhat_test  = mean(yhat_test_all,2);

%% ---------------- 5. Metrics (manual) ----------------
rmse_train = sqrt(mean((ytrain - yhat_train).^2));
mae_train  = mean(abs(ytrain - yhat_train));
r2_train   = 1 - sum((ytrain - yhat_train).^2) / sum((ytrain - mean(ytrain)).^2);

rmse_test = sqrt(mean((ytest - yhat_test).^2));
mae_test  = mean(abs(ytest - yhat_test));
r2_test   = 1 - sum((ytest - yhat_test).^2) / sum((ytest - mean(ytest)).^2);

fprintf('\n==== TRAIN ====\nRMSE: %.3f | MAE: %.3f | R²: %.3f\n', rmse_train, mae_train, r2_train);
fprintf('==== TEST  ====\nRMSE: %.3f | MAE: %.3f | R²: %.3f\n', rmse_test, mae_test, r2_test);

%% ---------------- 6. Variable Importance (manual variance method) ----------------
imp = zeros(size(Xtrain,2),1);
for j = 1:size(Xtrain,2)
    imp(j) = var(Xtrain(:,j) .* ytrain);
end
imp = imp / sum(imp); % normalize to sum=1
[impSorted, order] = sort(imp,'descend');
labels = predictors(order);

figure('Color','w');
bar(impSorted);
set(gca,'XTickLabel',labels,'XTickLabelRotation',30);
ylabel('Relative Importance (Variance Proxy)');
title('Variable Importance – Basic Ensemble (No Toolbox)');
grid on;
saveas(gcf,'fig_variable_importance_basic_final.png');

%% ---------------- 7. Actual vs Predicted ----------------
figure('Color','w');
plot(dateTest, ytest, 'o-','DisplayName','Actual'); hold on;
plot(dateTest, yhat_test, 's-','DisplayName','Predicted');
xlabel('Month'); ylabel('Burned Area (ha)');
title('Actual vs Predicted (Test Set)');
legend('Location','best'); grid on;
saveas(gcf,'fig_actual_vs_pred_basic_final.png');

fprintf('\nAll finished. No toolbox used. Images saved:\n');
fprintf(' - fig_variable_importance_basic_final.png\n');
fprintf(' - fig_actual_vs_pred_basic_final.png\n');
