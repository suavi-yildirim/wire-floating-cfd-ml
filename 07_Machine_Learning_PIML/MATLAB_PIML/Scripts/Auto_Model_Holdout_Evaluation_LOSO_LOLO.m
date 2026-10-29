%% =========================================================================
%  UNIFIED AUTOMATED HOLDOUT EVALUATION (LOLO & LOSO)
%  Automates surrogate model training and generates publication-grade 4-panel
%  figures for Flow Rate and Supply Pressure without Regression Learner (This processed is automated).
%  Supports:
%    - LOLO (Leave-One-Length-Out) : e.g., 2.0 m unseen length
%    - LOSO (Leave-One-Section-Out): e.g., 1.5 mm^2 unseen cross-section
%  Dynamic Encoding:
%    - If Length Holdout : color = Wire Cross-Section (5-step gradient)
%    - If Section Holdout: color = Wire Length (4-step gradient)
%    - Tube Diameter is always encoded by marker size (WC08 vs WC10)
%% =========================================================================
clc; clear; close all;
rng(1);

%% ---------------------------- CONFIGURATION -------------------------------
% Set Holdout Partitioning:
%   HOLDOUT_TYPE : 'Length' (LOLO) or 'Section' (LOSO)
HOLDOUT_TYPE    = 'Section';                      
HOLDOUT_VAL     =  6;                           % e.g., 2.0 wire length, 1.5 Cross-section

% Target Execution Mode: 'Both', 'Flow', or 'Pressure'
TARGET_MODE     = 'Both';                        

% Model Selection (trained automatically on training rows)
MODEL_FLOW      = 'Coarse Tree';                 % Top model for Flow (RMSE 1.08 Nm^3/h)
MODEL_PRESS     = 'Squared Exponential GPR';     % Top model for Pressure

% Evaluation & Display Options
N_BOOT          = 2000;                          % Bootstrap resamples for 95% CIs
PRESSURE_UNIT   = 'mbar';                        % 'mbar' or 'Pa'

% Academic Canvas Dimensions & Typography
FIG_SIZE_IN     = [1 1 15.0 8.2];                
TICK_FONT_SIZE  = 9.5;                           
LABEL_FONT_SIZE = 10.5;                          
TITLE_SIZE      = 11.5;                          
SGTITLE_SIZE    = 14;                            
LEGEND_SIZE     = 9.5;     
REF_LINE_WIDTH  = 1.2;                         

baseDir = fileparts(fileparts(mfilename('fullpath')));
allData = fullfile(baseDir, 'MAT_Dataset_v8', 'ML_Ready_Tables', 'ML_Data_All_v8.mat');
resDir  = fullfile(baseDir, 'Model_Evaluation_Results');
if ~exist(resDir, 'dir'), mkdir(resDir); end

%% ---------------------------- MASTER COLOR DEFINITIONS --------------------
FAPS_RGB = [106 138  34;     %  1 FAPS-Gruen dunkel
             63  95  68;     %  2 Gruen dunkel
              0  67  89;     %  3 Tuerkis dunkel
              4  30  66;     %  4 FAU-Blau dunkel
            149 162 171;     %  5 Grau 1
            255 203   0;     %  6 Sonderfarbe Gelb
            151 193  57;     %  7 FAPS-Gruen
            101 141 103;     %  8 Hausfarbe Gruen
             52 103 125;     %  9 Hausfarbe Tuerkis
              0  47 108;     % 10 FAU-Blau
            176 188 196;     % 11 Grau 2
            245 130  31]/255;% 12 Sonderfarbe Orange

LENGTHS_M    = [0.5 1.0 2.0 3.5];
SECTIONS_MM2 = [0.5 1.0 1.5 4.0 6.0];                              
TUBE_MM      = [8 10];
TUBE_DIAM_PT = [7.5 10];
TUBE_SIZE    = TUBE_DIAM_PT.^2;                  % Scatter area proportional to tube size
MS           = 77;                               % Marker size for legend keys

%% ---------------------------- HIGH-CONTRAST PALETTES ----------------------
% 1. WIRE CROSS-SECTION COLORS (SECTIONS_MM2 = [0.5, 1.0, 1.5, 4.0, 6.0] mm^2)
% 0.5: Orange | 1.0: Blue | 1.5: Gold/Yellow | 4.0: Purple | 6.0: Red
SEC_RGB = [245 130  31;      % 0.5 mm^2 -> Vivid Orange
    31 119 180;      % 1.0 mm^2 -> Classic Blue
    230 175   0;      % 1.5 mm^2 -> Saturated Gold / Yellow
    130  50 160;      % 4.0 mm^2 -> Deep Purple
    215  35  35]/255; % 6.0 mm^2 -> Bright Red

% 2. WIRE LENGTH COLORS (LENGTHS_M = [0.5, 1.0, 2.0, 3.5] m)
% 0.5: Blue | 1.0: Gold/Yellow | 2.0: Orange | 3.5: Red
LEN_RGB = [  31 119 180;      % 0.5 m -> Blue
    230 175   0;      % 1.0 m -> Saturated Gold / Yellow
    245 130  31;      % 2.0 m -> Orange
    215  35  35]/255; % 3.5 m -> Red

% --- FLOW THEME (GREEN FAMILY) ---
Pal_Flow.Len_RGB = LEN_RGB;
Pal_Flow.Sec_RGB = SEC_RGB;
Pal_Flow.cModel  = [151 193  57]/255; % FAPS-Gruen (Açık Yeşil: Model çubuğu)[cite: 1]
Pal_Flow.cPhys   = [106 138  34]/255; % FAPS-Gruen dunkel (Koyu Yeşil: Fizik çubuğu)[cite: 1]
Pal_Flow.cHist   = [151 193  57]/255; % FAPS-Gruen (Histogram dolgusu)[cite: 1]
Pal_Flow.cParity = [0 0 0];           % Saf siyah kesikli referans çizgisi

% --- PRESSURE THEME (BLUE FAMILY) ---
Pal_Press.Len_RGB = LEN_RGB;
Pal_Press.Sec_RGB = SEC_RGB;
Pal_Press.cModel  = [ 65 130 190]/255; % FAU-Blau açık tonu (Model çubuğu)
Pal_Press.cPhys   = [  0  47 108]/255; % FAU-Blau kurumsal tonu (Fizik çubuğu)[cite: 1]
Pal_Press.cHist   = [ 65 130 190]/255; % FAU-Blau açık tonu (Histogram dolgusu)
Pal_Press.cParity = [0 0 0];           % Saf siyah kesikli referans çizgisi
%% ---------------------------- DATA LOAD & SPLIT ---------------------------
fprintf('Loading master dataset from: %s\n', allData);
if exist(allData, 'file') ~= 2
    error('Dataset file not found: %s', allData);
end
S_all = load(allData);
T_all = S_all.ML_Data_All;

% Ensure all geometric and physics predictors are populated
T_all = prepareFeatures(T_all);

% Dynamic holdout partitioning
switch lower(HOLDOUT_TYPE)
    case 'length'
        isTest = abs(T_all.Length_m - HOLDOUT_VAL) < 1e-6;
        splitDesc = sprintf('Length_%.1fm', HOLDOUT_VAL);
        sliceName = sprintf('Unseen %.1f m Wire', HOLDOUT_VAL);
    case 'section'
        isTest = abs(T_all.Section_mm2 - HOLDOUT_VAL) < 1e-6;
        splitDesc = sprintf('Section_%.1fmm2', HOLDOUT_VAL);
        sliceName = sprintf('Unseen %.1f mm^2 Wire', HOLDOUT_VAL);
    otherwise
        error('Invalid HOLDOUT_TYPE: %s. Use ''Length'' or ''Section''.', HOLDOUT_TYPE);
end

TrainTable = T_all(~isTest, :);
TestTable  = T_all(isTest, :);

uids = unique(TestTable.WIRE_uid);
nW   = numel(uids);
fprintf('Partition [%s]: %d train runs | %d test runs (%d wire UIDs: %s)\n', ...
        splitDesc, height(TrainTable), height(TestTable), nW, mat2str(uids'));

%% ---------------------------- EXECUTION QUEUE -----------------------------
tasks = {};
if strcmpi(TARGET_MODE, 'Both') || strcmpi(TARGET_MODE, 'Flow')
    tasks{end+1} = struct('Var', 'Flow_Nm3h', 'Disp', 'Flow', 'Unit', 'Nm^3/h', ...
                          'Phys', 'Flow_Phys_Nm3h', 'Model', MODEL_FLOW, ...
                          'Scale', 1.0, 'Palette', Pal_Flow, ...
                          'Preds', {{'Section_mm2', 'Ferrule_to_Tube_Ratio', 'Blockage_Ratio', 'Length_over_Dh', 'Flow_Phys_Nm3h'}});
end
if strcmpi(TARGET_MODE, 'Both') || strcmpi(TARGET_MODE, 'Pressure')
    pScale = 1.0;
    pUnit  = 'Pa';
    if strcmpi(PRESSURE_UNIT, 'mbar')
        pScale = 1/100;                          % Pa -> mbar
        pUnit  = 'mbar';
    end
    tasks{end+1} = struct('Var', 'Pressure_Pa', 'Disp', 'Pressure', 'Unit', pUnit, ...
                          'Phys', 'Pressure_Phys_Darcy_Pa', 'Model', MODEL_PRESS, ...
                          'Scale', pScale, 'Palette', Pal_Press, ...
                          'Preds', {{'Section_mm2', 'Tube_D_mm', 'Length_m', 'Blockage_Ratio', ...
                                     'Length_over_Dh', 'Ferrule_to_Tube_Ratio', ...
                                     'Pressure_Phys_Darcy_Pa', 'Sys_Dynamic_Pa'}});
end

%% ---------------------------- MODELING & VISUALIZATION LOOP ---------------
for tIdx = 1:numel(tasks)
    tsk = tasks{tIdx};
    Pal = tsk.Palette;
    fprintf('\n============================================================\n');
    fprintf(' EVALUATING: %s (Model: %s on %s)\n', tsk.Disp, tsk.Model, splitDesc);
    fprintf('============================================================\n');

    % 1) Automatic Model Training
    tTrain = tic;
    predictFcn = trainSurrogate(TrainTable, tsk.Var, tsk.Preds, tsk.Model);
    fprintf('  Model training completed in %.2f s\n', toc(tTrain));

    % 2) Generate Predictions
    y     = TestTable.(tsk.Var) * tsk.Scale;
    yhat  = predictFcn(TestTable) * tsk.Scale;
    yphys = TestTable.(tsk.Phys) * tsk.Scale;

    [okT, kT] = ismember(TestTable.Tube_D_mm, TUBE_MM);

    % Dynamic Encoding Setup (Section vs Length holdout)
    if strcmpi(HOLDOUT_TYPE, 'Length')
        % In a length holdout, color points by Wire Cross-Section
        [~, kGroup] = ismember(round(TestTable.Section_mm2, 2), round(SECTIONS_MM2, 2));
        groupRGB    = Pal.Sec_RGB;
        groupVals   = SECTIONS_MM2;
        groupUnit   = 'mm^2';
        groupAxisLabel = 'Wire Cross-Section [mm^2]';
        groupFilter = @(k) abs(TestTable.Section_mm2 - groupVals(k)) < 1e-6;
        legendFmt   = '%.1f mm^2';
    else
        % In a section holdout, color points by Wire Length
        [~, kGroup] = ismember(round(TestTable.Length_m, 1), LENGTHS_M);
        groupRGB    = Pal.Len_RGB;
        groupVals   = LENGTHS_M;
        groupUnit   = 'm';
        groupAxisLabel = 'Wire Length [m]';
        groupFilter = @(k) abs(TestTable.Length_m - groupVals(k)) < 1e-6;
        legendFmt   = '%.1f m';
    end

    % 3) Metrics
    mModel = metricsOf(y, yhat);
    mPhys  = metricsOf(y, yphys);
    fprintf('  Validation Metrics:\n');
    fprintf('    %-14s R2: %7.4f | RMSE: %8.4f %s | MAE: %8.4f %s | Bias: %+8.4f %s\n', ...
            [tsk.Model ' (ML)'], mModel.R2, mModel.RMSE, tsk.Unit, mModel.MAE, tsk.Unit, mModel.bias, tsk.Unit);
    fprintf('    %-14s R2: %7.4f | RMSE: %8.4f %s | MAE: %8.4f %s | Bias: %+8.4f %s\n', ...
            'Physics Baseline', mPhys.R2, mPhys.RMSE, tsk.Unit, mPhys.MAE, tsk.Unit, mPhys.bias, tsk.Unit);

    % 4) Bootstrap Confidence Intervals
    rowsOf = arrayfun(@(u) find(TestTable.WIRE_uid == u), uids, 'UniformOutput', false);
    bR2   = nan(N_BOOT, 1);
    bRMSE = nan(N_BOOT, 1);
    for b = 1:N_BOOT
        pick = randi(nW, nW, 1);
        idx  = vertcat(rowsOf{pick});
        mb   = metricsOf(y(idx), yhat(idx));
        bR2(b)   = mb.R2;
        bRMSE(b) = mb.RMSE;
    end
    sR2 = sort(bR2); sRM = sort(bRMSE);
    ciR2   = [sR2(round(0.025*N_BOOT)), sR2(round(0.975*N_BOOT))];
    ciRMSE = [sRM(round(0.025*N_BOOT)), sRM(round(0.975*N_BOOT))];
    fprintf('    95%% CIs: R^2 [%.3f, %.3f] | RMSE [%.4f, %.4f] %s\n', ...
            ciR2(1), ciR2(2), ciRMSE(1), ciRMSE(2), tsk.Unit);

    % 5) Panel B Group-level RMSE Breakdown
    presentGroups = unique(groupVals(ismember(groupVals, groupVals(unique(kGroup(kGroup > 0))))));
    nGrp = numel(presentGroups);
    rmseModel = zeros(nGrp, 1);
    rmsePhys  = zeros(nGrp, 1);
    groupLabels = cell(nGrp, 1);
    for k = 1:nGrp
        idx = groupFilter(k);
        if any(idx)
            rmseModel(k) = sqrt(mean((yhat(idx) - y(idx)).^2));
            rmsePhys(k)  = sqrt(mean((yphys(idx) - y(idx)).^2));
        end
        groupLabels{k} = sprintf(legendFmt, presentGroups(k));
    end

    % 6) Render 4-Panel Figure
    rs       = yhat - y;                             
    raw_min  = min([y; yhat]);
    raw_max  = max([y; yhat]);
    data_rng = raw_max - raw_min;
    pad      = 0.08 * data_rng;                      % 8% margin (use 0.05 for 5%, 0.10 for 10%)
    lims     = [raw_min - pad, raw_max + pad];

    tag  = sprintf('%s_%s_%s', tsk.Disp, splitDesc, strrep(tsk.Model, ' ', '_'));

    fig = figure('Name', tag, 'Color', 'w', 'Units', 'inches', 'Position', FIG_SIZE_IN);
    if isprop(fig, 'Theme'), fig.Theme = 'light'; end
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'normal');

    % Panel A: Parity Plot
    axA = nexttile(tl); hold(axA, 'on');
    plotByGroup(axA, y, yhat, kGroup, kT, groupRGB, TUBE_SIZE);
    plot(axA, lims, lims, '--', 'Color', Pal.cParity, 'LineWidth', REF_LINE_WIDTH, 'HandleVisibility', 'off');

    xr   = lims(2) - lims(1);
    xMid = mean(lims);
    yk   = lims(1) + 0.055*xr;                   
    dx   = 0.14*xr;                              
    xOffsets = ((1:numel(TUBE_MM)) - mean(1:numel(TUBE_MM))) * dx - 0.025*xr;
    for it = 1:numel(TUBE_MM)
        xi = xMid + xOffsets(it);
        scatter(axA, xi, yk, TUBE_SIZE(it), [1 1 1], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
        text(axA, xi + 0.018*xr, yk, sprintf('WC%02d', TUBE_MM(it)), ...
             'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', ...
             'Color', 'k', 'FontSize', LEGEND_SIZE);
    end
    hold(axA, 'off');
    xlim(axA, lims); ylim(axA, lims); grid(axA, 'on');
    xlabel(axA, sprintf('Measured %s (%s)', tsk.Disp, tsk.Unit));
    ylabel(axA, sprintf('Predicted %s (%s)', tsk.Disp, tsk.Unit));
    title(axA, sprintf('A) Parity Plot: Predicted vs. Measured at Levitation Equilibrium', tsk.Model));
    text(axA, 0.04, 0.94, sprintf('R^2 = %.3f\nRMSE = %.3f %s', mModel.R2, mModel.RMSE, tsk.Unit), ...
         'Units', 'normalized', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
         'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 5, 'Color', 'k', ...
         'FontSize', LEGEND_SIZE);

    % Panel B: Sub-group RMSE Bar Comparison
    axB = nexttile(tl);
    bh = bar(axB, 1:nGrp, [rmseModel, rmsePhys]);
    bh(1).FaceColor = Pal.cModel;
    bh(2).FaceColor = Pal.cPhys;
    for j = 1:2
        bh(j).EdgeColor = 'k';
        bh(j).LineWidth = 0.6;
    end
    set(axB, 'XTick', 1:nGrp, 'XTickLabel', groupLabels);
    xlabel(axB, groupAxisLabel);
    ylabel(axB, sprintf('Holdout RMSE vs. Measured (%s)', tsk.Unit));

    if strcmpi(HOLDOUT_TYPE, 'Length')
         subGroupType = 'Cross-Section';
    else
         subGroupType = 'Length';
    end
         title(axB, sprintf('B) Holdout RMSE across %s Subgroups', subGroupType));

legB = legend(axB, {sprintf('%s (ML)', tsk.Model), 'Physics Baseline'}, ...
                  'Location', 'northwest', 'Box', 'off');
    legB.TextColor = 'k';
    grid(axB, 'on');

    maxBar = max([rmseModel(:); rmsePhys(:)]);
    ylim(axB, [0, maxBar * 1.25]);                   % 25% headroom for legend clearance

    % Panel C: Residuals vs. Measured
    axC = nexttile(tl); hold(axC, 'on');
    plotByGroup(axC, y, rs, kGroup, kT, groupRGB, TUBE_SIZE);
    yline(axC, 0, '--', 'Color', [0 0 0], 'LineWidth', 1.2, 'HandleVisibility', 'off');

    MS_MID = 77;  % ~76.6 pt^2
    
    hLegScatter = gobjects(numel(presentGroups), 1);
    for ig = 1:numel(presentGroups)                  
        hLegScatter(ig) = scatter(axC, -9999, -9999, MS_MID, groupRGB(ig,:), 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 0.8, ...
                'DisplayName', sprintf(legendFmt, presentGroups(ig)));
    end
    hold(axC, 'off');
    grid(axC, 'on');

    xlim(axC, lims);                                
    rsRange = max(rs) - min(rs);
    ylim(axC, [min(rs) - 0.15*rsRange, max(rs) + 0.35*rsRange]);

    lgdC = legend(axC, hLegScatter, 'Location', 'north', 'Orientation', 'horizontal', ...
                  'NumColumns', numel(presentGroups), 'Box', 'off');
    xlabel(axC, sprintf('Measured %s (%s)', tsk.Disp, tsk.Unit));
    ylabel(axC, sprintf('Residual, Predicted - Measured (%s)', tsk.Unit));
    title(axC, sprintf('C) %s Residuals at Levitation Equilibrium', tsk.Disp));

    % Panel D: Residual Distribution & Bias
    axD = nexttile(tl);
    hHist = histogram(axD, rs, 15, 'FaceColor', Pal.cHist, 'EdgeColor', 'k', 'LineWidth', 0.6);
    hold(axD, 'on');
    xline(axD, 0, '--', 'Color', [0 0 0], 'LineWidth', 1.2, 'HandleVisibility', 'off');
    hold(axD, 'off');
    
    maxDev = max(abs(rs));
    xlim(axD, [-1 * maxDev, 1 * maxDev]);

    maxCount = max(hHist.Values);
    ylim(axD, [0, maxCount * 1.30]);

   grid(axD, 'on');
    xlabel(axD, sprintf('Residual, Predicted - Measured (%s)', tsk.Unit));
    ylabel(axD, 'Frequency');
    title(axD, 'D) Residual Distribution & Bias');
    text(axD, 0.97, 0.92, sprintf('mean = %+.3f %s', mean(rs), tsk.Unit), ...
     'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
     'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 5, 'Color', 'k', ...
     'FontSize', LEGEND_SIZE);

    % Super-Title
   % Super-Title
    if strcmpi(HOLDOUT_TYPE, 'Length')
        sliceDetail = sprintf('n = %d runs across %d wire sections', height(TestTable), nW);
    else
        sliceDetail = sprintf('n = %d runs across %d wire lengths', height(TestTable), nW);
    end
    
    st = sgtitle(tl, sprintf('%s Prediction at Stable Levitation: %s on %s (%s)', ...
            tsk.Disp, tsk.Model, sliceName, sliceDetail), ...
            'FontSize', SGTITLE_SIZE, 'FontWeight', 'bold', 'Color', 'k');

    % Styling
    for ax = [axA, axB, axC, axD]
        ax.Color = 'w'; ax.XColor = 'k'; ax.YColor = 'k';
        ax.Box = 'on'; ax.LineWidth = 0.6; ax.FontSize = TICK_FONT_SIZE;
        ax.Title.FontSize = TITLE_SIZE; ax.Title.FontWeight = 'bold'; ax.Title.Color = 'k';
        ax.XLabel.FontSize = LABEL_FONT_SIZE; ax.XLabel.FontWeight = 'bold'; ax.XLabel.Color = 'k';
        ax.YLabel.FontSize = LABEL_FONT_SIZE; ax.YLabel.FontWeight = 'bold'; ax.YLabel.Color = 'k';
        ax.GridColor = [0.85 0.85 0.85]; ax.GridAlpha = 0.7;
    end
    lgdC.FontSize = LEGEND_SIZE; lgdC.TextColor = 'k';
    legB.FontSize = LEGEND_SIZE;

  
    drawnow;
    try
        
        for k = 1:numel(lgdC.EntryContainer.NodeChildren)
            iconObj = lgdC.EntryContainer.NodeChildren(k).Icon.Transform.Children.Children;
            if isprop(iconObj, 'SizeData')
                iconObj.SizeData = MS_MID;
            end
        end
    catch
       
    end

    % 7) Export Artifacts to Model_Evaluation_Results
    figPDF = fullfile(resDir, ['Fig_' tag '.pdf']);
    figPNG = fullfile(resDir, ['Fig_' tag '.png']);
    exportgraphics(fig, figPDF, 'ContentType', 'vector', 'BackgroundColor', 'white');
    exportgraphics(fig, figPNG, 'Resolution', 300, 'BackgroundColor', 'white');

    Summary = table([string(tsk.Model); "Physics Baseline"], ...
                    [mModel.R2; mPhys.R2], [mModel.RMSE; mPhys.RMSE], ...
                    [mModel.MAE; mPhys.MAE], [mModel.bias; mPhys.bias], ...
                    'VariableNames', {'Predictor','R2','RMSE','MAE','Bias'});
    writetable(Summary, fullfile(resDir, ['Summary_' tag '.csv']));

    Out = table(TestTable.Source_File, TestTable.Section_mm2, TestTable.Length_m, ...
                TestTable.Tube_D_mm, TestTable.WIRE_uid, TestTable.Trajectory_History, ...
                y, yhat, yphys, ...
                'VariableNames', {'Source_File','Section_mm2','Length_m','Tube_D_mm', ...
                                  'WIRE_uid','Trajectory_History','Measured','Predicted_Model','Predicted_Physics'});
    writetable(Out, fullfile(resDir, ['Predictions_' tag '.csv']));

    fprintf('  Saved publication artifacts:\n    -> %s\n    -> %s\n', figPDF, figPNG);
end

%% =========================================================================
%  LOCAL HELPER FUNCTIONS
%% =========================================================================
function predictFcn = trainSurrogate(Tr, respVar, predNames, modelChoice)
    switch modelChoice
        case 'Coarse Tree'
            mdl = fitrtree(Tr, respVar, 'PredictorNames', predNames, ...
                           'MinLeafSize', 36, 'Prune', 'off');
            predictFcn = @(T) predict(mdl, T);
        case 'Medium Tree'
            mdl = fitrtree(Tr, respVar, 'PredictorNames', predNames, ...
                           'MinLeafSize', 12, 'Prune', 'off');
            predictFcn = @(T) predict(mdl, T);
        case 'Fine Tree'
            mdl = fitrtree(Tr, respVar, 'PredictorNames', predNames, ...
                           'MinLeafSize', 4, 'Prune', 'off');
            predictFcn = @(T) predict(mdl, T);
        case 'Boosted Trees'
            t_template = templateTree('MinLeafSize', 8);
            mdl = fitrensemble(Tr, respVar, 'Method', 'LSBoost', ...
                               'NumLearningCycles', 100, 'Learners', t_template, ...
                               'LearnRate', 0.1, 'PredictorNames', predNames);
            predictFcn = @(T) predict(mdl, T);
        case 'Squared Exponential GPR'
            mdl = fitrgp(Tr, respVar, 'PredictorNames', predNames, ...
                         'KernelFunction', 'squaredexponential', 'Standardize', true);
            predictFcn = @(T) predict(mdl, T);
        otherwise
            error('Unknown model choice: %s', modelChoice);
    end
end

function T = prepareFeatures(T)
    if ~ismember('Section_mm2', T.Properties.VariableNames) && ismember('WIRE_section', T.Properties.VariableNames)
        T.Section_mm2 = T.WIRE_section;
    end
    if ~ismember('Length_m', T.Properties.VariableNames) && ismember('WIRE_lenght_m', T.Properties.VariableNames)
        T.Length_m = T.WIRE_lenght_m;
    end
    if ~ismember('Blockage_Ratio', T.Properties.VariableNames) && ismember('Wire_OD_mm', T.Properties.VariableNames)
        T.Blockage_Ratio = T.Wire_OD_mm ./ T.Tube_D_mm;
    end
    if ~ismember('Length_over_Dh', T.Properties.VariableNames) && ismember('Length_m', T.Properties.VariableNames)
        Dh_mm = T.Tube_D_mm - T.Wire_OD_mm;
        T.Length_over_Dh = (T.Length_m * 1000) ./ Dh_mm;
    end
    if ~ismember('Ferrule_to_Tube_Ratio', T.Properties.VariableNames)
        ferrArea = zeros(height(T), 1);
        for i = 1:height(T)
            df = getFerruleOD(T.Wire_OD_mm(i));
            ferrArea(i) = pi/4 * df^2;
        end
        T.Ferrule_to_Tube_Ratio = ferrArea ./ (pi/4 * T.Tube_D_mm.^2);
    end

    if ~ismember('Sys_Dynamic_Pa', T.Properties.VariableNames) || ...
       ~ismember('Pressure_Phys_Darcy_Pa', T.Properties.VariableNames) || ...
       ~ismember('Flow_Phys_Nm3h', T.Properties.VariableNames)
        
        R_AIR    = 287.058;
        P_AMB_PA = 101325;
        RHO_REF  = P_AMB_PA / (R_AIR * 273.15);
        MU_AIR   = 1.7894e-5;
        CD_FIX   = 0.88;
        CF_START = 1.145;
        
        rho_op = P_AMB_PA / (R_AIR * mean(T.Temp_K));
        [Q_phys, P_phys, sys_dyn] = physBaseline(T, rho_op, RHO_REF, MU_AIR, CD_FIX, CF_START);
        
        T.Flow_Phys_Nm3h           = Q_phys;
        T.Pressure_Phys_Darcy_Pa   = P_phys;
        T.Sys_Dynamic_Pa           = sys_dyn;
    end
end

function [Qnm3h, dpDarcy, dynSys, Re] = physBaseline(T, rho, RHO_REF, MU_AIR, cd, cf)
    P_AMB_PA = 101325;
    d        = T.Wire_OD_mm / 1000;
    Dh       = T.Hydraulic_Diameter_mm / 1000;
    Dt       = T.Tube_D_mm / 1000;
    Af       = T.Flow_Area_mm2 * 1e-6;
    At       = pi/4 * Dt.^2;
    
    d_ferrule = zeros(size(d));
    for i = 1:numel(d)
        d_ferrule(i) = getFerruleOD(T.Wire_OD_mm(i)) / 1000;
    end
    
    Aend = pi/4 * d_ferrule.^2;
    L    = T.Length_m;
    W    = T.Wire_Weight_N;

    v = sqrt(2 * W ./ (rho * cd .* Aend));
    for it = 1:500
        Re_it = rho * v .* Dh / MU_AIR;
        Cf_it = cf * 0.079 * Re_it.^(-0.25);
        vn    = sqrt(2 * W ./ (rho * (cd .* Aend + Cf_it .* pi .* d .* L)));
        if max(abs(vn - v) ./ vn) < 1e-10
            v = vn; break;
        end
        v = vn;
    end

    Re      = rho * v .* Dh / MU_AIR;
    f       = 0.316 * Re.^(-0.25);
    dpDarcy = P_AMB_PA + (f .* (L ./ Dh) .* rho .* v.^2 / 2);
    Qnm3h   = 3600 * (rho * Af .* v) / RHO_REF;
    dynSys  = 0.5 * rho * (Af .* v ./ At).^2;
end

function plotByGroup(ax, x, z, kGroup, kT, groupRGB, tubeSize)
    for ig = 1:size(groupRGB, 1)
        for it = 1:numel(tubeSize)
            idx = kGroup == ig & kT == it;
            if ~any(idx), continue; end
            scatter(ax, x(idx), z(idx), tubeSize(it), groupRGB(ig,:), 'filled', ...
                    'MarkerEdgeColor', 'k', 'LineWidth', 0.4, 'HandleVisibility', 'off');
        end
    end
end

function m = metricsOf(y, yh)
    e      = yh - y;                              
    m.R2   = 1 - sum(e.^2)/sum((y - mean(y)).^2);
    m.RMSE = sqrt(mean(e.^2));
    m.MAE  = mean(abs(e));
    m.bias = mean(e);
end

function od_f = getFerruleOD(od_wire)
    if abs(od_wire - 2.1) < 0.1
        od_f = 2.9;
    elseif abs(od_wire - 2.4) < 0.1
        od_f = 3.5;
    elseif abs(od_wire - 2.75) < 0.1
        od_f = 4.4;
    elseif abs(od_wire - 3.95) < 0.1
        od_f = 5.4;
    elseif abs(od_wire - 4.5) < 0.1
        od_f = 6.5;
    else
        od_f = od_wire;
    end
end