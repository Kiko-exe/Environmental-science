%% Step 0. 清理环境
clear; clc; close all;

%% Step 1. 读入数据（把你的 CSV 放在当前工作目录）
burned = readtable('burned area.csv');          % date_month (yyyy-MM), area_ha
rhcloud = readtable('RH and cloud.csv');        % time_utc, RH_pct, Cloud_pct
tempT   = readtable('temp.csv');                % time_utc, T2m_C, Td_C
windPR  = readtable('wind and rain.csv');       % time_utc, WS_ms, PR_mmph

%% Step 2. 解析时间字段
% burned 是月度字符串 'yyyy-MM'，转为月初日期便于 join
burned.date_month = datetime(burned.date_month,'InputFormat','yyyy-MM','TimeZone','UTC');
burned.Month = dateshift(burned.date_month,'start','month');
burned.MonthStr = datestr(burned.Month,'yyyy-mm');

% 其余是小时级 time_utc 'yyyy-MM-dd HH:mm'
rhcloud.time_utc = datetime(rhcloud.time_utc,'InputFormat','yyyy-MM-dd HH:mm','TimeZone','UTC');
tempT.time_utc   = datetime(tempT.time_utc,  'InputFormat','yyyy-MM-dd HH:mm','TimeZone','UTC');
windPR.time_utc  = datetime(windPR.time_utc, 'InputFormat','yyyy-MM-dd HH:mm','TimeZone','UTC');

%% Step 3. 转为 timetable 并按月聚合
tt_rh  = table2timetable(rhcloud,'RowTimes','time_utc');
tt_tmp = table2timetable(tempT,  'RowTimes','time_utc');
tt_wr  = table2timetable(windPR, 'RowTimes','time_utc');

% 按月聚合：风速/温度/湿度/云量取"月均"，降雨量是"月累积"
% 先均值
tt_rh_m  = retime(tt_rh, 'monthly','mean');      % RH_pct, Cloud_pct
tt_tmp_m = retime(tt_tmp,'monthly','mean');      % T2m_C, Td_C
tt_wr_m_mean = retime(tt_wr,'monthly','mean');   % WS_ms(均值), PR_mmph(均值，不是我们要的汇总)

% 再做降雨"月累积"（mm per hour 累和）
tt_wr_m_sum  = retime(tt_wr(:,'PR_mmph'),'monthly','sum');

% 合并气象月表
tt_met_m = synchronize(tt_rh_m, tt_tmp_m, tt_wr_m_mean(:,'WS_ms'), tt_wr_m_sum, 'intersection');

% 变回 table 并建 Month 键
met_m = timetable2table(tt_met_m, 'ConvertRowTimes', true);
met_m.Properties.VariableNames{1} = 'Month';
met_m.MonthStr = datestr(met_m.Month,'yyyy-mm');

%% Step 4. 与 burned area（已是月度）外连接
T = outerjoin(burned(:,{'Month','MonthStr','area_ha'}), ...
              met_m(:,{'Month','MonthStr','RH_pct','Cloud_pct','T2m_C','Td_C','WS_ms','PR_mmph'}), ...
              'Keys','Month','MergeKeys',true,'Type','left');

% 也可以只保留双方都有的月份
T = rmmissing(T,'DataVariables',{'area_ha','WS_ms','RH_pct','T2m_C','PR_mmph','Cloud_pct'});

% 为了可读，按时间排序
T = sortrows(T,'Month');

%% Step 5. 简要检查
disp(head(T,8));
writetable(T, 'merged_monthly_dataset.csv'); % 保存用于留档或复现
