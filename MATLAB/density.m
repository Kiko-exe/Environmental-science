%% ===========================================
%  Make Two Core Plots from 704.xlsx (No Toolbox)
%  Fig1: Burned Area vs Wind Speed (by tail/cross/head)
%  Fig2: Mean Burned Area by Tree Density × Base Spread Probability
%  Expected columns (exact):
%   'Wind','Wind Speed','Tree Density (%)','Base Spread Probability (%)','result'
% ============================================

clear; clc;

file = '704.xlsx';
if ~exist(file,'file'), error('704.xlsx not found in current folder.'); end
T = readtable(file);

% --- Column names from your sheet
colWind   = 'Wind';                        % words: north, southwest, ...
colSpeed  = 'WindSpeed';                  % numeric
colDensity= 'TreeDensity';            % numeric (30/60/90)
colBase   = 'BaseSpreadProbability'; % numeric (0/30/60/90)
colBurn   = 'result';                      % numeric = Burned Area (%)

% --- Coerce to numeric where needed
numify = @(x) str2double(string(x));
for c = {colSpeed,colDensity,colBase,colBurn}
    c = c{1};
    if ~isnumeric(T.(c)), T.(c) = numify(T.(c)); end
end

% --- Map wind words -> degrees, then → tail/cross/head relative to East (90°)
T.wind_deg = arrayfun(@parse_wind_word, string(T.(colWind)));
desired = 90; % East
dtheta  = mod(T.wind_deg - desired + 180, 360) - 180; % [-180,180]
absd    = abs(dtheta);
lab = strings(height(T),1);
lab(absd <= 45)              = "tailwind";
lab(absd > 45 & absd <=135)  = "crosswind";
lab(absd > 135)              = "headwind";
T.sector3 = categorical(lab, ["tailwind","crosswind","headwind"]);

% Keep valid rows
valid = ~isnan(T.wind_deg) & ~isnan(T.(colSpeed)) & ~isnan(T.(colBurn)) ...
      & ~isnan(T.(colDensity)) & ~isnan(T.(colBase));
T = T(valid,:);

%% ------------ FIG 1: Burned Area vs Wind Speed (by sector3) ------------
cats = categories(T.sector3);
markers = {'o','s','^'}; % different markers per sector
figure('Color','w'); box on; grid on; hold on;

legendEntries = strings(0);
for i = 1:numel(cats)
    idx = T.sector3 == cats{i};
    x = T.(colSpeed)(idx); y = T.(colBurn)(idx);
    scatter(x, y, 28, 'filled', markers{i}, 'MarkerFaceAlpha', 0.65);
    % simple linear fit (polyfit) if enough points
    [slope,intercept,r2,n] = simple_fit(x,y);
    if n>=3
        xs = linspace(min(x), max(x), 100);
        ys = slope*xs + intercept;
        plot(xs, ys, 'LineWidth', 1.6);
        legendEntries(end+1) = sprintf('%s  (R^2=%.2f, n=%d)', cats{i}, r2, n); %#ok<SAGROW>
    else
        legendEntries(end+1) = sprintf('%s  (n=%d)', cats{i}, n); %#ok<SAGROW>
    end
end
xlabel('Wind Speed'); ylabel('Burned Area (%)');
title('Burned Area vs Wind Speed by Wind Sector (tail / cross / head)');
legend(legendEntries, 'Location','best');
saveas(gcf, 'fig1_burn_vs_windspeed_by_sector.png'); close;

%% --- Helper: group mean & SE (standard error)
function [m,se] = grp_mean_se(v)
    v = v(~isnan(v));
    n = numel(v);
    if n==0, m=NaN; se=NaN; else, m=mean(v); se=std(v)/sqrt(n); end
end

%% ------------ FIG 2: Mean Burned Area: Density × Base Spread Prob ------------
dens_levels = unique(T.(colDensity)); dens_levels = dens_levels(~isnan(dens_levels));
base_levels = unique(T.(colBase));    base_levels = base_levels(~isnan(base_levels));
dens_levels = sort(dens_levels(:)');
base_levels = sort(base_levels(:)');

% Pre-allocate mean and SE matrix: rows=density, cols=base
M  = NaN(numel(dens_levels), numel(base_levels));
SE = NaN(size(M));

for ii = 1:numel(dens_levels)
    for jj = 1:numel(base_levels)
        idx = T.(colDensity)==dens_levels(ii) & T.(colBase)==base_levels(jj);
        [m,se] = grp_mean_se(T.(colBurn)(idx));
        M(ii,jj)  = m;
        SE(ii,jj) = se;
    end
end

% Plot as multi-line with error bars
figure('Color','w'); box on; grid on; hold on;
cols = lines(numel(base_levels)); % default colormap
for jj = 1:numel(base_levels)
    y  = M(:,jj);
    ye = SE(:,jj);
    % density on x-axis
    errorbar(dens_levels, y, ye, '-o', 'LineWidth',1.6, 'MarkerFaceColor', cols(jj,:));
end
xlabel('Tree Density (%)');
ylabel('Mean Burned Area (%)');
title('Mean Burned Area by Tree Density × Base Spread Probability');
legtxt = arrayfun(@(b) sprintf('Base Spread Prob = %g%%', b), base_levels, 'UniformOutput', false);
legend(legtxt, 'Location','northwest');
saveas(gcf, 'fig2_mean_burn_density_by_baseprob.png'); close;

%% ============== Local functions =================
function deg = parse_wind_word(w)
    % Map wind direction words to degrees (N=0, NE=45, E=90, SE=135, S=180, SW=225, W=270, NW=315)
    s = lower(strtrim(w));
    s = strrep(s,'-',''); s = strrep(s,' ',''); s = strrep(s,'_','');
    switch s
        case {'n','north'},      deg = 0;
        case {'ne','northeast'}, deg = 45;
        case {'e','east'},       deg = 90;
        case {'se','southeast'}, deg = 135;
        case {'s','south'},      deg = 180;
        case {'sw','southwest'}, deg = 225;
        case {'w','west'},       deg = 270;
        case {'nw','northwest'}, deg = 315;
        otherwise, deg = NaN;
    end
end

function [slope,intercept,r2,n] = simple_fit(x,y)
    % Linear fit via polyfit + R^2
    valid = ~(isnan(x)|isnan(y)); x=x(valid); y=y(valid);
    n = numel(x);
    if n<3, slope=NaN; intercept=NaN; r2=NaN; return; end
    p = polyfit(x,y,1); slope=p(1); intercept=p(2);
    yhat = polyval(p,x);
    ss_res = sum((y - yhat).^2);
    ss_tot = sum((y - mean(y)).^2);
    r2 = 1 - ss_res/ss_tot;
end
