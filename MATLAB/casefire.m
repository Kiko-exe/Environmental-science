%% =============================================
%  Wildfire Plots from 704.xlsx (exact headers)
%  Columns expected:
%  Scenario ID | Tree Density (%) | Base Spread Probability (%) | Wind | Wind Speed | big jump | Repetitions | r1..r5 | result | t1..t5 | ticks
%  Outputs:
%    fig1_burn_vs_density.png
%    fig2_burn_vs_windspeed.png
%    fig3_ticks_vs_burn.png
%    fig4_burn_vs_baseprob.png
%    fig5_burn_by_windsector.png (if Wind present)
%    summary_linear_fits.csv
%% =============================================

clear; clc;

file = '704.xlsx';

% 读取第一个工作表（如需指定工作表，可加 'Sheet', 1/ 'SheetName'）
T = readtable(file);

% ===== 精确映射列名（与你提供的完全一致）=====
colScenario   = 'Scenario ID';
colDensity    = 'TreeDensity';
colBaseProb   = 'BaseSpreadProbability';
colWind       = 'Wind';          % 方向（度），若为空则不出风向图
colWindSpeed  = 'WindSpeed';
colBurn       = 'result';        % 你的表里"result" = 汇总的 burned area (%)
colTicks      = 'ticks';

% 转换为数值
numify = @(x) str2double(string(x));
for c = {colDensity,colBaseProb,colWindSpeed,colWind,colBurn,colTicks}
    c = c{1};
    if ismember(c, T.Properties.VariableNames) && ~isnumeric(T.(c))
        T.(c) = numify(T.(c));
    end
end

% ====== 工具函数（不用 fitlm）======
function [slope,intercept,r2,n] = simple_fit(x,y)
    valid = ~(isnan(x)|isnan(y));
    x = x(valid); y = y(valid);
    n = numel(x);
    if n<3
        slope=NaN; intercept=NaN; r2=NaN; return;
    end
    % polyfit 拟合一阶直线
    p = polyfit(x,y,1);
    slope = p(1); intercept = p(2);
    % 预测和 R²
    yhat = polyval(p,x);
    ss_res = sum((y - yhat).^2);
    ss_tot = sum((y - mean(y)).^2);
    r2 = 1 - ss_res/ss_tot;
end

function make_plot(x,y,xlab,ylab,ttl,outpng,slope,intercept,r2)
    valid = ~(isnan(x)|isnan(y));
    x = x(valid); y = y(valid);
    if numel(x)<3, return; end
    figure('Color','w'); box on; grid on; hold on;
    scatter(x,y,24,'filled','MarkerFaceAlpha',0.6);
    xx = linspace(min(x),max(x),100);
    yy = slope*xx + intercept;
    plot(xx,yy,'r-','LineWidth',1.5);
    xlabel(xlab); ylabel(ylab);
    title(sprintf('%s\nLinear fit R^2=%.2f',ttl,r2));
    saveas(gcf,outpng); close;
    fprintf('Saved %s (R^2=%.3f, n=%d)\n',outpng,r2,numel(x));
end

% 摘要表
sumRows = {};

% 图1：Burned Area vs Tree Density
[slope,intercept,r2,n] = simple_fit(T.(colDensity),T.(colBurn));
make_plot(T.(colDensity),T.(colBurn),'Tree Density (%)','Burned Area (%)',...
    'Burned Area vs Tree Density','fig1_burn_vs_density.png',slope,intercept,r2);
sumRows(end+1,:) = {'fig1_burn_vs_density.png',colDensity,colBurn,slope,intercept,r2,n};

% 图2：Burned Area vs Wind Speed
[slope,intercept,r2,n] = simple_fit(T.(colWindSpeed),T.(colBurn));
make_plot(T.(colWindSpeed),T.(colBurn),'Wind Speed','Burned Area (%)',...
    'Burned Area vs Wind Speed','fig2_burn_vs_windspeed.png',slope,intercept,r2);
sumRows(end+1,:) = {'fig2_burn_vs_windspeed.png',colWindSpeed,colBurn,slope,intercept,r2,n};

% 图3：Ticks vs Burned Area
[slope,intercept,r2,n] = simple_fit(T.(colBurn),T.(colTicks));
make_plot(T.(colBurn),T.(colTicks),'Burned Area (%)','Ticks',...
    'Ticks vs Burned Area','fig3_ticks_vs_burn.png',slope,intercept,r2);
sumRows(end+1,:) = {'fig3_ticks_vs_burn.png',colBurn,colTicks,slope,intercept,r2,n};

% 图4：Burned Area vs Base Spread Probability
[slope,intercept,r2,n] = simple_fit(T.(colBaseProb),T.(colBurn));
make_plot(T.(colBaseProb),T.(colBurn),'Base Spread Probability (%)','Burned Area (%)',...
    'Burned Area vs Base Spread Probability','fig4_burn_vs_baseprob.png',slope,intercept,r2);
sumRows(end+1,:) = {'fig4_burn_vs_baseprob.png',colBaseProb,colBurn,slope,intercept,r2,n};

% 图5：按风向分扇区（如果有 wind）
if ismember(colWind,T.Properties.VariableNames) && any(~isnan(T.(colWind)))
    ang = mod(T.(colWind),360);
    edges = 0:45:360; labels = {'N','NE','E','SE','S','SW','W','NW'};
    bin = discretize(ang,edges);
    bin(isnan(bin))=1; bin(bin==numel(edges))=1;
    wind_sector = categorical(labels(bin));
    valid = ~isundefined(wind_sector) & ~isnan(T.(colBurn));
    if any(valid)
        figure('Color','w'); box on; grid on;
        boxchart(wind_sector(valid),T.(colBurn)(valid));
        xlabel('Wind Direction Sector'); ylabel('Burned Area (%)');
        title('Burned Area by Wind Direction Sector');
        saveas(gcf,'fig5_burn_by_windsector.png'); close;
    end
end

% 保存摘要为 CSV
summary = cell2table(sumRows,'VariableNames',{'figure','x','y','slope','intercept','r2','n'});
writetable(summary,'summary_linear_fits.csv');
disp('Saved summary_linear_fits.csv');