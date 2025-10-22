%% ===========================================
% Make two core plots from 704.xlsx (no toolboxes)
% Fig1: Risk matrix = P(Burned >= THR) by Wind Sector (tail/cross/head) × Wind Speed
% Fig2: Mean Burned Area by Tree Density × Base Spread Probability (with SE)
% Expected columns (exact):
%  'Wind' (north/south/...), 'Wind Speed', 'Tree Density (%)',
%  'Base Spread Probability (%)', 'result' (Burned Area %)
%% ===========================================

clear; clc;
file = '704.xlsx';
if ~exist(file,'file'), error('704.xlsx not found in current folder.'); end
T = readtable(file);

% ---- exact column names
colWind    = 'Wind';
colSpeed   = 'WindSpeed';
colDensity = 'TreeDensity';
colBase    = 'BaseSpreadProbability';
colBurn    = 'result';

% ---- numeric coercion
numify = @(x) str2double(string(x));
for c = {colSpeed,colDensity,colBase,colBurn}
    c = c{1};
    if ~isnumeric(T.(c)), T.(c) = numify(T.(c)); end
end

% ---- map wind words -> degrees, then -> tail/cross/head relative to East(90°)
T.wind_deg = arrayfun(@parse_wind_word, string(T.(colWind)));
desired = 90;                                           % West -> East
dtheta  = mod(T.wind_deg - desired + 180, 360) - 180;   % [-180,180]
absd    = abs(dtheta);
lab = strings(height(T),1);
lab(absd <= 45)              = "tailwind";
lab(absd > 45 & absd <=135)  = "crosswind";
lab(absd > 135)              = "headwind";
T.sector3 = categorical(lab, ["tailwind","crosswind","headwind"]);

% ---- keep valid
valid = ~isnan(T.wind_deg) & ~isnan(T.(colSpeed)) & ~isnan(T.(colBurn)) & ...
        ~isnan(T.(colDensity)) & ~isnan(T.(colBase));
T = T(valid,:);

%% ===================== FIG 1: 风向×风速的高烧毁概率 =====================
% 1) 风速分级：若风速值很多，按5取整；否则用原有离散水平
s_raw = T.(colSpeed);
speeds_unique = unique(s_raw(~isnan(s_raw)));
if numel(speeds_unique) > 6
    s_grp = round(s_raw/5)*5;
else
    s_grp = s_raw;
end
T.speed_bin = s_grp;
speed_levels = sort(unique(T.speed_bin(~isnan(T.speed_bin)))');

% 2) 计算每个格子的 P(result >= THR) 和样本数
THR = 50; % 阈值，可改为 60/70 做稳健性检验
sectors = categories(T.sector3);
P = nan(numel(sectors), numel(speed_levels));
N = zeros(size(P));
for i = 1:numel(sectors)
    for j = 1:numel(speed_levels)
        idx = (T.sector3==sectors{i}) & (T.speed_bin==speed_levels(j));
        y = T.(colBurn)(idx);
        N(i,j) = sum(idx);
        if N(i,j)>0
            P(i,j) = mean(y >= THR);
        end
    end
end

% 3) 画热力图 + 叠文字（百分比与n）
figure('Color','w'); box on;
imagesc(speed_levels, 1:numel(sectors), P*100);
set(gca,'YTick',1:numel(sectors),'YTickLabel',sectors);
xlabel('Wind Speed'); ylabel('Wind Sector vs East (tail / cross / head)');
title(sprintf('Probability of large burn (\\ge %d%%) by wind sector × speed', THR));
cb = colorbar; cb.Label.String = '% of runs';
caxis([0 100]);
% overlay text
for i = 1:numel(sectors)
    for j = 1:numel(speed_levels)
        if ~isnan(P(i,j))
            txt = sprintf('%d%%\\n(n=%d)', round(P(i,j)*100), N(i,j));
            text(speed_levels(j), i, txt, 'Color','w', 'FontSize',9, ...
                 'HorizontalAlignment','center', 'FontWeight','bold');
        end
    end
end
saveas(gcf,'fig1_prob_large_burn_heatmap.png'); close;

%% ================== FIG 2: 密度×基础传播概率（均值±SE） ==================
dens_levels = sort(unique(T.(colDensity)(~isnan(T.(colDensity))))');
base_levels = sort(unique(T.(colBase)(~isnan(T.(colBase))))');

% 计算均值与标准误
M  = nan(numel(dens_levels), numel(base_levels));
SE = nan(size(M));
for ii = 1:numel(dens_levels)
    for jj = 1:numel(base_levels)
        idx = (T.(colDensity)==dens_levels(ii)) & (T.(colBase)==base_levels(jj));
        y = T.(colBurn)(idx);
        y = y(~isnan(y));
        if ~isempty(y)
            M(ii,jj)  = mean(y);
            SE(ii,jj) = std(y)/sqrt(numel(y));
        end
    end
end

figure('Color','w'); box on; grid on; hold on;
cols = lines(numel(base_levels));
for jj = 1:numel(base_levels)
    errorbar(dens_levels, M(:,jj), SE(:,jj), '-o', 'LineWidth',1.6, ...
        'MarkerFaceColor', cols(jj,:));
end
xlabel('Tree Density (%)'); ylabel('Mean Burned Area (%)');
title('Mean Burned Area by Tree Density × Base Spread Probability');
legtxt = arrayfun(@(b) sprintf('Base Spread Prob = %g%%', b), base_levels, 'UniformOutput', false);
legend(legtxt, 'Location','northwest');
saveas(gcf,'fig2_mean_burn_density_by_baseprob.png'); close;

%% ====================== local helpers ======================
function deg = parse_wind_word(w)
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
