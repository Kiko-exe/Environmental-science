% ========== Base Spread Probability × Variable → Percent Burned (colored legends, no ticks) ==========
clear; clc; close all;

filePath = '704.xlsx';
outPNG   = 'spread_bivariate_panels_color.png';

% ---- read (keep real names with spaces) ----
opts = detectImportOptions(filePath, 'PreserveVariableNames', true);
T = readtable(filePath, opts);

% required columns
need = {'Tree Density','Base Spread Probability','Wind','Wind Speed','result'};
for k = 1:numel(need)
    if ~ismember(need{k}, T.Properties.VariableNames)
        error('Missing column: %s', need{k});
    end
end

% target y = percent burned area (%)
y = T.('result');
if all(isnan(y)), error('"result" is all NaN.'); end
yf = y(~isnan(y));
if ~isempty(yf) && all(yf>=0 & yf<=1), y = y*100; end
yLabel = 'percent burned area (%)';

% spread levels
S = T.('Base Spread Probability');
if ~isnumeric(S), error('"Base Spread Probability" must be numeric (e.g., 0,30,60,90).'); end
spreadLevels = sort(unique(S(~isnan(S))))';
nSpread = numel(spreadLevels);

% panels (3 only, ticks removed)
xNames  = {'Tree Density','Wind Speed','Wind'};
pTitles = {'Tree Density × Spread', 'Wind Speed × Spread', 'Wind × Spread'};

% color/marker consistent across panels (one color per spread)
colors = lines(max(nSpread,4));
mks    = {'o','s','^','d','v','>','<'};

% global y-limits (1–99 percentile)
yok = y(~isnan(y)); 
yl = [min(prctile(yok,1)), max(prctile(yok,99))];
if ~isfinite(yl(1)) || ~isfinite(yl(2)), yl = [min(yok) max(yok)]; end
if yl(1)==yl(2), yl = yl + [-1 1]; end

% figure
F = figure('Color','w','Position',[70 70 1500 900]);
tlo = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
sgtitle('Base Spread Probability × Variable  →  percent burned area (%)','FontWeight','bold','FontSize',14);

for p = 1:numel(xNames)
    nexttile; hold on; grid on;
    xname = xNames{p}; title(pTitles{p},'FontWeight','bold');
    xlabel(xname); ylabel(yLabel);

    xcol = T.(xname);
    isCat = iscellstr(xcol) || isstring(xcol) || iscategorical(xcol);

    % legend handles (ensure colored legend)
    hLegend = gobjects(1,nSpread);
    legTxt  = cell(1,nSpread);

    if isCat
        xcat = categorical(xcol);
        cats = categories(xcat);
        xpos = 1:numel(cats);
        set(gca,'XTick',xpos,'XTickLabel',cats,'XTickLabelRotation',30);

        for si = 1:nSpread
            sp  = spreadLevels(si);
            sub = T(S==sp, :);
            if isempty(sub), continue; end

            xsub = categorical(sub.(xname), cats);
            ysub = sub.('result');
            ysf  = ysub(~isnan(ysub));
            if ~isempty(ysf) && all(ysf>=0 & ysf<=1), ysub = ysub*100; end

            [~,~,idx] = unique(xsub);
            mu = NaN(1,numel(cats)); se = NaN(1,numel(cats));
            for c = 1:numel(cats)
                sel = (idx==c) & ~isnan(ysub);
                if any(sel)
                    yy = ysub(sel);
                    mu(c) = mean(yy);
                    se(c) = std(yy)/sqrt(numel(yy));
                end
            end

            % main line (capture handle for legend color)
            h = plot(xpos, mu, '-', 'Color', colors(si,:), 'LineWidth',1.6, ...
                'Marker', mks{mod(si-1,numel(mks))+1}, 'MarkerSize',6);
            % error bars (not in legend)
            for c = 1:numel(xpos)
                if ~isnan(mu(c)) && ~isnan(se(c))
                    line([xpos(c) xpos(c)], [mu(c)-se(c), mu(c)+se(c)], 'Color', colors(si,:), 'LineWidth',1, ...
                        'HandleVisibility','off');
                end
            end
            hLegend(si) = h; legTxt{si} = sprintf('Spread = %g%%', sp);
        end

    else
        % numeric X
        xnum    = xcol;
        xlevels = sort(unique(xnum(~isnan(xnum))))';

        for si = 1:nSpread
            sp  = spreadLevels(si);
            sub = T(S==sp, :);
            if isempty(sub), continue; end

            xv = sub.(xname);
            yv = sub.('result');
            yvf = yv(~isnan(yv));
            if ~isempty(yvf) && all(yvf>=0 & yvf<=1), yv = yv*100; end

            mu = NaN(size(xlevels)); se = NaN(size(xlevels));
            for k = 1:numel(xlevels)
                sel = (xv==xlevels(k)) & ~isnan(yv);
                if any(sel)
                    yy = yv(sel);
                    mu(k) = mean(yy);
                    se(k) = std(yy)/sqrt(numel(yy));
                end
            end

            h = plot(xlevels, mu, '-', 'Color', colors(si,:), 'LineWidth',1.6, ...
                'Marker', mks{mod(si-1,numel(mks))+1}, 'MarkerSize',6);
            for k = 1:numel(xlevels)
                if ~isnan(mu(k)) && ~isnan(se(k))
                    line([xlevels(k) xlevels(k)], [mu(k)-se(k), mu(k)+se(k)], 'Color', colors(si,:), ...
                        'LineWidth',1, 'HandleVisibility','off');
                end
            end
            hLegend(si) = h; legTxt{si} = sprintf('Spread = %g%%', sp);
        end
    end

    ylim(yl);

    % build a colored legend using handles (filters invalid)
    valid = isgraphics(hLegend);
    legend(hLegend(valid), legTxt(valid), 'Location','northwest');
end

% save
exportgraphics(F, outPNG, 'Resolution', 220);
fprintf('Saved: %s\n', outPNG);