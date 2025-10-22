% ================== Percent Burned Area vs Predictors (clean) ==================
% 需求：
% - 目标列：result（在图中显示为 "percent burned area (%)"）
% - 排除：r1..r5、t1..t5、repetition、scenario_id
% - 画 result 与其它所有数值变量的一对一关系（散点 + 线性拟合），标题含 r / p / N
% - 零工具箱：p 值基于 betainc 的 t 分布实现

clear; clc; close all;

% ----- 设置 -----
filePath = '704.xlsx';     % 读取当前目录
saveFig  = true;
outMain  = 'percent_burned_vs_all.png';
outRank  = 'percent_burned_corr_rank.png';

% ----- 读表并清洗 -----
opts = detectImportOptions(filePath, 'PreserveVariableNames', true);
T = readtable(filePath, opts);
T = standardizeMissing(T, {NaN,'NA','Na','','null','Null'});

% 变量名统一为下划线小写，便于匹配
origNames = string(T.Properties.VariableNames);
normNames = lower(regexprep(origNames,'\W+','_'));
T.Properties.VariableNames = cellstr(normNames);

% ----- 仅保留数值列，去掉全 NaN -----
isNum = varfun(@(x) isnumeric(x), T, 'OutputFormat','uniform');
Tnum  = T(:, isNum);
if isempty(Tnum.Properties.VariableNames)
    error('未找到数值型变量。');
end
keep = true(1, width(Tnum));
for i = 1:width(Tnum)
    if all(isnan(Tnum.(i))), keep(i) = false; end
end
Tnum = Tnum(:, keep);

% ----- 目标列：result -> y（若为 0~1 比例，转为百分比） -----
if ~ismember('result', Tnum.Properties.VariableNames)
    error('必须存在数值列 "result"。');
end
y = Tnum.result;
yf = y(~isnan(y));
if ~isempty(yf) && all(yf >= 0 & yf <= 1)
    y = y * 100;
end
yLabel = 'percent burned area (%)';

% ----- 构建自变量集合：剔除重复试次与指定列 -----
rAll = {'r1','r2','r3','r4','r5'};
tAll = {'t1','t2','t3','t4','t5'};
dropList = [rAll, tAll, {'result','repetitions','scenario_id'}];

predNames = string(Tnum.Properties.VariableNames);
predNames = predNames(~ismember(predNames, dropList));
p = numel(predNames);
if p == 0
    error('没有可用于与 "result" 作图的自变量（都被排除了）。');
end

% 如不想把 ticks 作为自变量，请取消下一行注释：
% predNames = predNames(~strcmp(predNames,'ticks'));

% ----- 计算每个自变量与 result 的 r、p、N（零工具箱） -----
rVals = NaN(1, p); pVals = NaN(1, p); nVals = zeros(1, p);
for i = 1:p
    xi = Tnum.(predNames(i));
    ok = ~isnan(xi) & ~isnan(y);
    n = sum(ok); nVals(i) = n;
    if n >= 3
        xv = xi(ok); yv = y(ok);
        mx = mean(xv); my = mean(yv);
        vx = xv - mx; vy = yv - my;
        denom = sqrt(sum(vx.^2) * sum(vy.^2));
        if denom > 0
            r = sum(vx .* vy) / denom;
            rVals(i) = r;
            df = n - 2;
            if df > 0 && abs(r) < 1
                tstat = r * sqrt(df / (1 - r^2));
                pVals(i) = two_sided_p_from_t(tstat, df);  % 无工具箱
            end
        end
    end
end

% ----- 作图：result vs 每个自变量 -----
ncol = min(2, p);
nrow = ceil(p / ncol);

F = figure('Color','w','Position',[80 60 1500 900]);
tlo = tiledlayout(nrow, ncol, 'TileSpacing','compact', 'Padding','compact');
sgtitle('percent burned area (%) vs predictors', 'FontWeight','bold', 'FontSize', 14);

% 统一 y 轴范围（1–99 分位）
yok = y(~isnan(y));
ymin = prctile(yok, 1); ymax = prctile(yok, 99);
if ~isfinite(ymin) || ~isfinite(ymax)
    ymin = min(yok); ymax = max(yok);
end
if ymin == ymax, ymin = ymin - 1; ymax = ymax + 1; end

for i = 1:p
    nexttile;
    xi = Tnum.(predNames(i));
    ok = ~isnan(xi) & ~isnan(y);
    xv = xi(ok); yv = y(ok);

    plot(xv, yv, 'o', 'MarkerSize', 4); grid on; hold on;
    xlabel(char(predNames(i)), 'Interpreter','none');
    ylabel(yLabel);
    ylim([ymin ymax]);

    if numel(xv) >= 2
        pf = polyfit(xv, yv, 1);
        xx = linspace(min(xv), max(xv), 100);
        yy = polyval(pf, xx);
        plot(xx, yy, '-', 'LineWidth', 1.5);
    end

    ri = rVals(i); pi = pVals(i); ni = nVals(i);
    if ~isnan(ri)
        if ~isnan(pi)
            ttl = sprintf('%s vs %s | r=%.2f, p=%.3g, N=%d', yLabel, char(predNames(i)), ri, pi, ni);
        else
            ttl = sprintf('%s vs %s | r=%.2f, N=%d', yLabel, char(predNames(i)), ri, ni);
        end
    else
        ttl = sprintf('%s vs %s | insufficient data', yLabel, char(predNames(i)));
    end
    title(ttl, 'Interpreter','none','FontWeight','bold','FontSize',10);
end

if saveFig
    exportgraphics(F, outMain, 'Resolution', 200);
    fprintf('Exported: %s\n', outMain);
end

% ----- |r| 排名柱状图（便于挑重点） -----
[~, ord] = sort(abs(rVals), 'descend', 'MissingPlacement','last');
G = figure('Color','w','Position',[120 80 900 500]);
bar(abs(rVals(ord))); grid on;
set(gca, 'XTick', 1:p, 'XTickLabel', cellstr(predNames(ord)), 'XTickLabelRotation',45);
ylabel('|r| with percent burned area (%)');
title('Predictors ranked by |correlation| with percent burned area (%)','FontWeight','bold');
if saveFig
    exportgraphics(G, outRank, 'Resolution', 200);
    fprintf('Exported: %s\n', outRank);
end

% ================= 本地函数：两侧 p 值（基于 t 分布，无工具箱） =================
function p = two_sided_p_from_t(t, v)
    % two-sided p-value for t-statistic t with v dof via incomplete beta
    z = v / (v + t^2);
    upper = 0.5 * betainc(z, v/2, 0.5);  % upper tail for |t|
    p = 2 * upper;
    if p > 1, p = 1; end
end
