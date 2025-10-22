%% ==================== Best Composite Figure for 704.xlsx (toolbox-free) ====================
clear; clc; close all;

filePath = '704.xlsx';
opts = detectImportOptions(filePath, 'PreserveVariableNames', true);
T = readtable(filePath, opts);
T = standardizeMissing(T, {NaN, "NA", "Na", "", "null", "Null"});
emptyCols = varfun(@(c) all(ismissing(c)), T, 'OutputFormat','uniform');
T(:, emptyCols) = [];

rawNames = string(T.Properties.VariableNames);
cleanNames = lower(regexprep(rawNames, '\W+', '_'));
T.Properties.VariableNames = cellstr(cleanNames);

% 识别日期列
dateColIdx = [];
for i = 1:width(T)
    if isdatetime(T.(i)), dateColIdx = i; break; end
    if iscellstr(T.(i)) || isstring(T.(i))
        tryDT = NaT(height(T),1);
        ok = false;
        try, tryDT = datetime(T.(i), 'InputFormat','yyyy-MM-dd', 'Format','yyyy-MM-dd'); ok = true; end
        if ~ok
            try, tryDT = datetime(T.(i)); ok = true; end
        end
        if ok && sum(~isnat(tryDT)) >= max(5, round(0.5*height(T)))
            T.(i) = tryDT; dateColIdx = i; break;
        end
    end
end
dateVec = []; if ~isempty(dateColIdx), dateVec = T.(dateColIdx); end

% 只取数值列
isNum = varfun(@(x) isnumeric(x), T, 'OutputFormat','uniform');
Tnum = T(:, isNum);
if isempty(Tnum.Properties.VariableNames), error('未找到数值型变量。'); end
keep = true(1, width(Tnum));
for i = 1:width(Tnum)
    xi = Tnum.(i);
    if ~isnumeric(xi) || all(isnan(xi)), keep(i) = false; end
end
Tnum = Tnum(:, keep);

varNames = string(Tnum.Properties.VariableNames);
X = Tnum{:,:};
nVar = size(X,2);

% 自动识别目标变量（burn/area/ba/burnt/burned）
targetIdx = [];
keyTokens = ["burn","area","ba","burnt","burned"];
for i = 1:numel(varNames)
    vn = lower(varNames(i));
    if any(contains(vn, keyTokens)), targetIdx = i; break; end
end

% -------- 改动点：手写 Pearson 相关系数（无 corr 函数） --------
R = NaN(nVar);
for i = 1:nVar
    xi = X(:,i);
    for j = 1:nVar
        yj = X(:,j);
        valid = ~isnan(xi) & ~isnan(yj);
        if sum(valid) >= 3
            xv = xi(valid); yv = yj(valid);
            % 计算 r = cov(x,y) / (std(x)*std(y))
            mx = mean(xv); my = mean(yv);
            vx = xv - mx; vy = yv - my;
            denom = sqrt(sum(vx.^2) * sum(vy.^2));
            if denom > 0
                R(i,j) = sum(vx .* vy) / denom;
            else
                R(i,j) = NaN; % 若某列方差为0
            end
        end
    end
end
% -------------------------------------------------------------

% 挑最强相关的三对（或 target 最相关的三对）
pairs = [];
for i = 1:nVar-1
    for j = i+1:nVar
        rij = R(i,j);
        if ~isnan(rij)
            pairs = [pairs; i, j, rij, abs(rij)]; %#ok<AGROW>
        end
    end
end

if ~isempty(targetIdx)
    isTargetPair = (pairs(:,1)==targetIdx) | (pairs(:,2)==targetIdx);
    TP = pairs(isTargetPair, :);
    if isempty(TP)
        [~,ord] = sort(pairs(:,4), 'descend');
        topPairs = pairs(ord(1:min(3,end)), :);
    else
        [~,ord] = sort(TP(:,4), 'descend'); TP = TP(ord,:);
        out = []; cnt = 0;
        for k = 1:size(TP,1)
            a = TP(k,1); b = TP(k,2);
            otherIdx = a; if a==targetIdx, otherIdx = b; end
            if otherIdx ~= targetIdx
                out = [out; TP(k,:)]; %#ok<AGROW>
                cnt = cnt+1; if cnt==3, break; end
            end
        end
        if isempty(out)
            [~,ord] = sort(pairs(:,4), 'descend');
            out = pairs(ord(1:min(3,end)), :);
        end
        topPairs = out;
    end
else
    if isempty(pairs)
        topPairs = [];
    else
        [~,ord] = sort(pairs(:,4), 'descend');
        topPairs = pairs(ord(1:min(3,end)), :);
    end
end

% =============== 画图：2x2 仪表盘 ==================
f = figure('Color','w','Position',[100 80 1400 900]);
tlo = tiledlayout(2,2, 'TileSpacing','compact', 'Padding','compact');
sgtitle('704.xlsx — Exploratory Dashboard (toolbox-free)', 'FontWeight','bold', 'FontSize',14);

% (A) 时间序列
nexttile;
if ~isempty(dateVec) && ~isempty(targetIdx)
    y = X(:, targetIdx);
    plot(dateVec, y, '-o', 'LineWidth',1.2, 'MarkerSize',3); grid on;
    xlabel('Date'); ylabel(varNames(targetIdx), 'Interpreter','none');
    title('(A) Target over Time', 'FontWeight','bold');
elseif ~isempty(dateVec)
    stds = std(X, 'omitnan');
    [~,mx] = max(stds);
    y = X(:, mx);
    plot(dateVec, y, '-o', 'LineWidth',1.2, 'MarkerSize',3); grid on;
    xlabel('Date'); ylabel(varNames(mx), 'Interpreter','none');
    title('(A) Most Variable Series over Time', 'FontWeight','bold');
else
    stds = std(X, 'omitnan');
    [~,mx] = max(stds);
    plot(1:size(X,1), X(:,mx), '-o', 'LineWidth',1.2, 'MarkerSize',3); grid on;
    xlabel('Index'); ylabel(varNames(mx), 'Interpreter','none');
    title('(A) Most Variable Series (Index)', 'FontWeight','bold');
end

% (B) 相关性热图
nexttile;
imagesc(R, [-1 1]); axis image; colormap(parula); colorbar;
title('(B) Correlation (pairwise, Pearson)', 'FontWeight','bold');
set(gca,'XTick',1:nVar, 'XTickLabel',varNames, 'XTickLabelRotation',45, ...
        'YTick',1:nVar, 'YTickLabel',varNames);
hold on;
for i = 1:nVar
    for j = 1:nVar
        if ~isnan(R(i,j))
            text(j, i, sprintf('%.2f', R(i,j)), ...
                 'HorizontalAlignment','center', 'Color','w', 'FontSize',8, 'FontWeight','bold');
        end
    end
end
hold off;

% (C) & (D) 最强关系散点+拟合线（手写 polyfit）
makeScatterWithFit = @(ax, x, y, nameX, nameY) local_scatter_fit(ax, x, y, nameX, nameY);

if ~isempty(topPairs)
    nexttile;
    i1 = topPairs(1,1); j1 = topPairs(1,2);
    makeScatterWithFit(gca, X(:,i1), X(:,j1), varNames(i1), varNames(j1));
    title(sprintf('(C) %s vs %s (r=%.2f)', varNames(j1), varNames(i1), R(i1,j1)), ...
        'FontWeight','bold','Interpreter','none');

    nexttile;
    idx2 = min(2, size(topPairs,1));
    i2 = topPairs(idx2,1); j2 = topPairs(idx2,2);
    makeScatterWithFit(gca, X(:,i2), X(:,j2), varNames(i2), varNames(j2));
    title(sprintf('(D) %s vs %s (r=%.2f)', varNames(j2), varNames(i2), R(i2,j2)), ...
        'FontWeight','bold','Interpreter','none');
else
    nexttile([1 2]);
    plotmatrix(X); title('(C/D) Scatterplot Matrix (fallback)', 'FontWeight','bold');
    set(gca,'FontSize',8);
end

% 另开窗口做全变量散点矩阵
figure('Color','w','Position',[100 80 900 900]);
plotmatrix(X);
sgtitle('Scatterplot Matrix — All Numeric Variables', 'FontWeight','bold');
set(gca,'FontSize',8);

% ================== 本地函数 ==================
function local_scatter_fit(ax, x, y, nameX, nameY)
    v = ~isnan(x) & ~isnan(y); x = x(v); y = y(v);
    axes(ax); hold(ax,'on');
    plot(ax, x, y, 'o', 'MarkerSize',4, 'HandleVisibility','off');
    grid(ax,'on'); xlabel(ax, nameX, 'Interpreter','none'); ylabel(ax, nameY, 'Interpreter','none');
    if numel(x) >= 2
        p = polyfit(x, y, 1);
        xx = linspace(min(x), max(x), 100);
        yy = polyval(p, xx);
        plot(ax, xx, yy, '-', 'LineWidth',1.5, ...
             'DisplayName', sprintf('fit: y = %.3fx %+ .3f', p(1), p(2)));
        legend(ax, 'Location','best');
    end
end
