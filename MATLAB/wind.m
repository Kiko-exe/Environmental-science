%% ============================================
% Analyze wind (cardinal words) & speed effects from 704.xlsx
% Uses only base MATLAB (no toolboxes).
% Expected columns (exact names from your file):
%   'Wind' (north/south/... words), 'Wind Speed', 'result' (burned area %), 'ticks'
% Target spread direction: West -> East (azimuth = 90°)
%% ============================================

clear; clc;

file = '704.xlsx';
if ~exist(file,'file'), error('704.xlsx not found in current folder.'); end
T = readtable(file);

% ----- Exact column names you gave -----
colWind   = 'Wind';
colSpeed  = 'WindSpeed';
colBurn   = 'result';
colTicks  = 'ticks';

% ----- Coerce numeric for used numeric columns -----
numify = @(x) str2double(string(x));
for c = {colSpeed,colBurn,colTicks}
    c = c{1};
    if ismember(c,T.Properties.VariableNames) && ~isnumeric(T.(c))
        T.(c) = numify(T.(c));
    end
end

% ----- Map wind words -> degrees (N=0, NE=45, E=90, SE=135, S=180, SW=225, W=270, NW=315) -----
if ~ismember(colWind,T.Properties.VariableNames)
    error('Column "%s" not found in table.', colWind);
end
T.wind_deg = arrayfun(@local_parse_wind_word, string(T.(colWind)));

% Keep only valid rows
valid = ~isnan(T.wind_deg) & ~isnan(T.(colSpeed)) & ~isnan(T.(colBurn)) & ~isnan(T.(colTicks));
T = T(valid,:);

% ----- Alignment with desired eastward spread (90°) -----
desired = 90;  % East
dtheta  = mod(T.wind_deg - desired + 180, 360) - 180;  % [-180,180]
T.align = cosd(dtheta);  % +1 tailwind (顺风), 0 crosswind, -1 headwind (逆风)

% ----- Sector labels from words (8 sectors) -----
T.sector8 = categorical(local_sector8_from_deg(T.wind_deg), ...
    {'N','NE','E','SE','S','SW','W','NW'});

% ----- Tail/Cross/Head 分类（按与东向差角） -----
absd = abs(dtheta);
lab = strings(height(T),1);
lab(absd <= 45)              = "tailwind";
lab(absd > 45 & absd <=135)  = "crosswind";
lab(absd > 135)              = "headwind";
T.sector3 = categorical(lab, ["tailwind","crosswind","headwind"]);

%% ---------- FIG A: wind words分布（极坐标） ----------
figure('Color','w');
polarhistogram(deg2rad(T.wind_deg), 16); hold on;
rl = rlim; polarplot([deg2rad(desired) deg2rad(desired)], [0 rl(2)], 'LineWidth',1.8);
title('Wind direction distribution (line = desired East 90°)');
saveas(gcf,'figA_wind_dirs_polar.png'); close;

%% ---------- FIG B: Burned Area vs Alignment ----------
[slope,intercept,r2,n] = local_simple_fit(T.align, T.(colBurn));
figure('Color','w'); box on; grid on; hold on;
scatter(T.align, T.(colBurn), 26,'filled','MarkerFaceAlpha',0.6);
xx = linspace(-1,1,100); yy = slope*xx + intercept; plot(xx,yy,'LineWidth',1.8);
xlabel('Alignment with East (cos(\Delta\theta))'); ylabel('Burned Area (%)');
title(sprintf('Burned Area vs Alignment (R^2=%.2f, n=%d)', r2, n));
saveas(gcf,'figB_burn_vs_alignment.png'); close;

%% ---------- FIG C: Ticks vs Alignment ----------
[slope,intercept,r2,n] = local_simple_fit(T.align, T.(colTicks));
figure('Color','w'); box on; grid on; hold on;
scatter(T.align, T.(colTicks), 26,'filled','MarkerFaceAlpha',0.6);
xx = linspace(-1,1,100); yy = slope*xx + intercept; plot(xx,yy,'LineWidth',1.8);
xlabel('Alignment with East (cos(\Delta\theta))'); ylabel('Ticks');
title(sprintf('Ticks vs Alignment (R^2=%.2f, n=%d)', r2, n));
saveas(gcf,'figC_ticks_vs_alignment.png'); close;

%% ---------- FIG D: Burned Area by 8 wind sectors ----------
figure('Color','w'); box on; grid on;
boxchart(T.sector8, T.(colBurn));
xlabel('Wind sector (8)'); ylabel('Burned Area (%)');
title('Burned Area by wind word sector (N, NE, ..., NW)');
saveas(gcf,'figD_burn_by_8sectors.png'); close;

%% ---------- FIG E: Burned Area vs Wind Speed within tail/cross/head ----------
cats = categories(T.sector3);
figure('Color','w'); 
for i = 1:numel(cats)
    idx = T.sector3 == cats{i};
    x = T.(colSpeed)(idx); y = T.(colBurn)(idx);
    [sl,ic,r2s,ns] = local_simple_fit(x,y);
    subplot(1,numel(cats),i);
    scatter(x,y,20,'filled'); hold on; box on; grid on;
    if ns>=3
        xs = linspace(min(x),max(x),100); ys = sl*xs + ic; plot(xs,ys,'LineWidth',1.5);
    end
    xlabel('Wind Speed'); ylabel('Burned Area (%)');
    title(sprintf('%s (R^2=%.2f, n=%d)', cats{i}, r2s, ns));
end
set(gcf,'Position',[100 100 1200 360]);
saveas(gcf,'figE_burn_vs_speed_by_sector3.png'); close;

%% ---------- FIG F: Heatmap of Burned Area (alignment × wind speed) ----------
nb1 = 10; nb2 = 10;
xb = linspace(-1,1,nb1+1);
yb = linspace(min(T.(colSpeed)), max(T.(colSpeed)), nb2+1);
H  = accumarray([discretize(T.align,xb), discretize(T.(colSpeed),yb)], T.(colBurn), [nb1,nb2], @mean, NaN);
figure('Color','w');
imagesc((xb(1:end-1)+xb(2:end))/2, (yb(1:end-1)+yb(2:end))/2, H');
set(gca,'YDir','normal'); colorbar; box on; grid on;
xlabel('Alignment with East'); ylabel('Wind Speed');
title('Mean Burned Area (%) by alignment × wind speed');
saveas(gcf,'figF_burn_heatmap_align_speed.png'); close;

%% ---------- Export summary stats (for report table) ----------
S = [];
% Overall: Burned ~ Alignment
[sA,iA,r2A,nA] = local_simple_fit(T.align, T.(colBurn));
S = [S; {"overall_burn_vs_alignment", sA, iA, r2A, nA}];

% Per sector3: Burned ~ Speed
for i = 1:numel(cats)
    idx = T.sector3 == cats{i};
    [sS,iS,r2S,nS] = local_simple_fit(T.(colSpeed)(idx), T.(colBurn)(idx));
    S = [S; {sprintf('%s_burn_vs_speed', cats{i}), sS, iS, r2S, nS}];
end

summary = cell2table(S, 'VariableNames', {'model','slope','intercept','r2','n'});
writetable(summary,'wind_models_summary.csv');
disp('Saved wind_models_summary.csv');

%% =================== Local functions ===================
function deg = local_parse_wind_word(w)
% Map wind direction words to azimuth degrees.
% Accepts many aliases (case-insensitive, spaces/hyphens ignored).
    s = lower(strtrim(w));
    s = strrep(s,'-',''); s = strrep(s,' ',''); s = strrep(s,'_','');
    switch s
        case {'n','north'},    deg = 0;
        case {'ne','northeast','north-east'}, deg = 45;
        case {'e','east'},     deg = 90;
        case {'se','southeast','south-east'}, deg = 135;
        case {'s','south'},    deg = 180;
        case {'sw','southwest','south-west'}, deg = 225;
        case {'w','west'},     deg = 270;
        case {'nw','northwest','north-west'}, deg = 315;
        otherwise, deg = NaN;  % unknown token
    end
end

function labs = local_sector8_from_deg(deg)
% Return 8-sector labels from degrees
    labs = strings(size(deg));
    for k = 1:numel(deg)
        d = mod(deg(k),360);
        if (d>=337.5 || d<22.5),  labs(k)="N";
        elseif d<67.5,            labs(k)="NE";
        elseif d<112.5,           labs(k)="E";
        elseif d<157.5,           labs(k)="SE";
        elseif d<202.5,           labs(k)="S";
        elseif d<247.5,           labs(k)="SW";
        elseif d<292.5,           labs(k)="W";
        else                      labs(k)="NW";
        end
    end
end

function [slope,intercept,r2,n] = local_simple_fit(x,y)
% Linear fit via polyfit (no toolboxes) + R^2
    valid = ~(isnan(x)|isnan(y));
    x = x(valid); y = y(valid);
    n = numel(x);
    if n<3, slope=NaN; intercept=NaN; r2=NaN; return; end
    p = polyfit(x,y,1);
    slope = p(1); intercept = p(2);
    yhat = polyval(p,x);
    ss_res = sum((y - yhat).^2);
    ss_tot = sum((y - mean(y)).^2);
    r2 = 1 - ss_res/ss_tot;
end
