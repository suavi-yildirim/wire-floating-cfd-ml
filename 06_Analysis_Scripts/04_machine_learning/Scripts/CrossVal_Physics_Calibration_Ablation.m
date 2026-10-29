%% =========================================================================
%  WFE STAGE 5 (v8): physics calibration, grouped cross-validation, ablation,
%                    residual analysis
%
%  Input : MAT_Dataset_v8/ML_Ready_Tables/ML_Data_All_v8.mat
%  Output: MAT_Dataset_v8/CV_Results_v8.mat, MAT_Dataset_v8/CV_OutOfFold_v8.csv
%
%  Grouped schemes (every row is predicted out-of-fold):
%    LOSO  leave-one-section-out : unseen wire cross-section (extrapolation)
%    LOUO  leave-one-UID-out     : unseen wire (length and weight), seen section
%    LOLO  leave-one-length-out  : unseen wire length (scaling & drag partitioning)
%
%  Inside each fold: CD and CF are fitted on training rows only; physics
%  baselines and surrogates are trained on training rows only; held-out rows
%  are predicted once.
%
%  Predictor lists match the Regression Learner sessions:
%    FLOW     : Section_mm2, Tube_D_mm, Length_m, Blockage_Ratio, Length_over_Dh,
%               Flow_Phys_Nm3h
%    PRESSURE : Section_mm2, Tube_D_mm, Length_m, Blockage_Ratio, Length_over_Dh,
%               Pressure_Phys_Darcy_Pa, Sys_Dynamic_Pa
%% =========================================================================
clc; clear; close all;
tAll = tic;
fprintf('=== WFE STAGE 5: calibration, grouped CV (LOSO, LOUO, LOLO), ablation ===\n');

%% ---------------------------- SETTINGS ------------------------------------
dataFile = fullfile(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'MAT_Dataset_v8'), 'ML_Ready_Tables', 'ML_Data_All_v8.mat');
outDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'Model_Evaluation_Results');
R_AIR    = 287.058;                    % J/(kg K), dry air
P_AMB_PA = 101325;                     % Pa
RHO_REF  = P_AMB_PA/(R_AIR*273.15);    % kg/m^3, flow sensor reference (0 degC)
MU_AIR   = 1.7894e-5;                  % Pa s
CD_FIX   = 0.88;                       % CFD-based pure ferrule form drag (fixed reference)
CF_START = 1.145;                      % CFD-based skin-friction factor (fixed reference from CFD median)
CD_LO    = 0.40;                       % lower bounds for the fits
CF_LO    = 0.80;

%% ---------------------------- STAGE 1: LOAD -------------------------------
fprintf('Stage 1/4: loading %s\n', dataFile);
S  = load(dataFile);
T0 = S.ML_Data_All;
n  = height(T0);
fprintf('  %d runs | %d sections | %d wire UIDs | %d lengths\n', n, ...
        numel(unique(T0.Section_mm2)), numel(unique(T0.WIRE_uid)), numel(unique(T0.Length_m)));

% Friction-to-form force ratio (geometry check on the dominance claim)
rho0 = P_AMB_PA/(R_AIR*mean(T0.Temp_K));
[~, ~, ~, Re0] = physBaseline(T0, rho0, RHO_REF, MU_AIR, CD_FIX, CF_START);
d0    = T0.Wire_OD_mm/1000;
ffRat = (CF_START*0.079*Re0.^(-0.25)) .* (pi.*d0.*T0.Length_m) ./ (pi/4.*d0.^2);
fprintf('  Friction/form force ratio (CD=%.2f, CF=%.3f): min %.2f | median %.2f | max %.2f\n', ...
        CD_FIX, CF_START, min(ffRat), median(ffRat), max(ffRat));

%% ---------------------------- STAGE 2: GROUPED CV -------------------------
schemes = {'LOSO', 'Section_mm2'; ...
           'LOUO', 'WIRE_uid'; ...
           'LOLO', 'Length_m'};
Res = struct();

% Previous Predictor definitions
% fQ  = {'Section_mm2','Tube_D_mm','Length_m','Blockage_Ratio','Length_over_Dh','Flow_Phys_Nm3h'};
% fP  = {'Section_mm2','Tube_D_mm','Length_m','Blockage_Ratio','Length_over_Dh', ...
%        'Pressure_Phys_Darcy_Pa','Sys_Dynamic_Pa'};

% FLOW: 5 dimensionless + physics features
fQ = {'Section_mm2', 'Ferrule_to_Tube_Ratio', 'Blockage_Ratio', 'Length_over_Dh', 'Flow_Phys_Nm3h'};

% PRESSURE: includes Tube_D_mm, Length_m, and Sys_Dynamic_Pa for system inlet pressure scaling
fP = {'Section_mm2', 'Tube_D_mm', 'Length_m', 'Blockage_Ratio', 'Length_over_Dh', ...
      'Ferrule_to_Tube_Ratio', 'Pressure_Phys_Darcy_Pa', 'Sys_Dynamic_Pa'};

fP0 = fP(1:end-1);                       % same, without Sys_Dynamic_Pa (ablation)

for sc = 1:size(schemes,1)
    scName = schemes{sc,1};
    grp    = schemes{sc,2};
    folds  = unique(T0.(grp));
    nF     = numel(folds);
    fprintf('\nStage 2/4 [%s]: %d folds, grouped by %s\n', scName, nF, grp);
    fprintf('  %-8s %-5s %-7s %-7s %-22s %-22s %-14s %-6s\n', ...
            'fold','n','CD','CF','R2 flow phys','R2 flow GPR','R2 press GPR','t[s]');

    r = struct();
    r.Q_cal    = nan(n,1);   % physics, CD and CF fitted per fold
    r.Q_fix    = nan(n,1);   % physics, CD = CD_FIX and CF = CF_START, no fit
    r.Q_nofric = nan(n,1);   % physics, CF = 0, CD fitted per fold
    r.P_cal    = nan(n,1);   % Darcy absolute pressure at calibrated drag velocity
    r.Q_LM     = nan(n,1);   % linear surrogate, flow
    r.Q_GP     = nan(n,1);   % GPR surrogate, flow
    r.P_LM     = nan(n,1);   % linear surrogate, pressure, with Sys_Dynamic_Pa
    r.P_LM0    = nan(n,1);   % linear surrogate, pressure, without Sys_Dynamic_Pa
    r.P_GP     = nan(n,1);   % GPR surrogate, pressure

    foldRows = cell(nF,1);

    for f = 1:nF
        tf     = tic;
        teMask = T0.(grp) == folds(f);
        Tr     = T0(~teMask,:);
        Te     = T0( teMask,:);

        % 1) training-fold operating density and coefficient fits
        rhoTr    = P_AMB_PA/(R_AIR*mean(Tr.Temp_K));
        [cd, cf] = calibrateCDCF(Tr, rhoTr, RHO_REF, MU_AIR, CD_LO, CF_LO, CF_START);
        cdNF     = calibrateCDOnly(Tr, rhoTr, RHO_REF, MU_AIR, CD_LO);

        % 2) physics baselines for training and held-out rows
        [QtrC, PtrC, SYStr] = physBaseline(Tr, rhoTr, RHO_REF, MU_AIR, cd, cf);
        [QteC, PteC, SYSte] = physBaseline(Te, rhoTr, RHO_REF, MU_AIR, cd, cf);
        QteFix              = physBaseline(Te, rhoTr, RHO_REF, MU_AIR, CD_FIX, CF_START);
        QteNF               = physBaseline(Te, rhoTr, RHO_REF, MU_AIR, cdNF, 0);

        Tr.Flow_Phys_Nm3h = QtrC;   Tr.Pressure_Phys_Darcy_Pa = PtrC;  Tr.Sys_Dynamic_Pa = SYStr;
        Te.Flow_Phys_Nm3h = QteC;   Te.Pressure_Phys_Darcy_Pa = PteC;  Te.Sys_Dynamic_Pa = SYSte;

        % 3) surrogates trained on training rows only
        mQ_LM  = fitlm(Tr, ['Flow_Nm3h ~ '    strjoin(fQ,  ' + ')]);
        mP_LM  = fitlm(Tr, ['Pressure_Pa ~ '  strjoin(fP,  ' + ')]);
        mP_LM0 = fitlm(Tr, ['Pressure_Pa ~ '  strjoin(fP0, ' + ')]);
        mQ_GP  = fitrgp(Tr, 'Flow_Nm3h',   'PredictorNames', fQ, ...
                        'KernelFunction', 'ardsquaredexponential', 'Standardize', true);
        mP_GP  = fitrgp(Tr, 'Pressure_Pa', 'PredictorNames', fP, ...
                        'KernelFunction', 'ardsquaredexponential', 'Standardize', true);

        % 4) out-of-fold predictions for the held-out rows
        r.Q_cal(teMask)    = QteC;
        r.Q_fix(teMask)    = QteFix;
        r.Q_nofric(teMask) = QteNF;
        r.P_cal(teMask)    = PteC;
        r.Q_LM(teMask)     = predict(mQ_LM,  Te);
        r.Q_GP(teMask)     = predict(mQ_GP,  Te);
        r.P_LM(teMask)     = predict(mP_LM,  Te);
        r.P_LM0(teMask)    = predict(mP_LM0, Te);
        r.P_GP(teMask)     = predict(mP_GP,  Te);

        % fold log
        rQphys = r2(Te.Flow_Nm3h, QteC);
        rQgp   = r2(Te.Flow_Nm3h, r.Q_GP(teMask));
        rPgp   = r2(Te.Pressure_Pa, r.P_GP(teMask));
        foldRows{f} = table(folds(f), nnz(teMask), cd, cf, cdNF, rQphys, rQgp, rPgp, toc(tf), ...
            'VariableNames', {'Fold','nTest','CD','CF','CD_noFric', ...
                              'R2_Q_phys','R2_Q_GPR','R2_P_GPR','t_s'});
        fprintf('  %-8g %-5d %-7s %-7s %-22.3f %-22.3f %-14.3f %-6.1f\n', ...
                folds(f), nnz(teMask), sprintf('%.2f', cd), sprintf('%.2f', cf), ...
                rQphys, rQgp, rPgp, toc(tf));
    end

    r.Log = vertcat(foldRows{:});
    Res.(scName) = r;
end

%% ---------------------------- STAGE 3: SUMMARY ----------------------------
fprintf('\nStage 3/4: out-of-fold summary (all %d rows, identical rows for every method)\n', n);
yQ = T0.Flow_Nm3h;
yP = T0.Pressure_Pa;

for sc = {'LOSO', 'LOUO', 'LOLO'}
    r = Res.(sc{1});
    fprintf('\n--- %s ---\n', sc{1});

    Cq = [metricRow('Physics, CD & CF fitted per fold',               yQ, r.Q_cal);
          metricRow(sprintf('Physics, CD=%.2f & CF=%.3f fixed', CD_FIX, CF_START), yQ, r.Q_fix);
          metricRow('Physics, CF=0 (form drag only)',                 yQ, r.Q_nofric);
          metricRow('Linear surrogate',                               yQ, r.Q_LM);
          metricRow('GPR surrogate',                                  yQ, r.Q_GP)];
    fprintf('FLOW (Flow_Nm3h)\n');
    disp(Cq);

    Cp = [metricRow('Darcy at drag velocity, calibrated (Abs)',       yP, r.P_cal);
          metricRow('Linear, without Sys_Dynamic_Pa',                 yP, r.P_LM0);
          metricRow('Linear, with Sys_Dynamic_Pa',                    yP, r.P_LM);
          metricRow('GPR surrogate',                                  yP, r.P_GP)];
    fprintf('PRESSURE (Pressure_Pa)\n');
    disp(Cp);

    fprintf('Fitted coefficients across folds: CD %.2f to %.2f | CF %.2f to %.2f\n', ...
            min(r.Log.CD), max(r.Log.CD), min(r.Log.CF), max(r.Log.CF));
end

%% ---------------------------- STAGE 4: RESIDUALS, SAVE --------------------
fprintf('\nStage 4/4: hysteresis residuals (LOUO GPR, measured - predicted) and save\n');
rP = yP - Res.LOUO.P_GP;
rQ = yQ - Res.LOUO.Q_GP;
Gt = table(T0.Trajectory_History, rP, rQ, ...
           'VariableNames', {'Trajectory_History','Res_Press_Pa','Res_Flow_Nm3h'});
disp(groupsummary(Gt, 'Trajectory_History', {'median','mean'}, ...
                  {'Res_Press_Pa','Res_Flow_Nm3h'}));
if exist('ranksum', 'file') == 2
    isR = T0.Trajectory_History == "Rises";
    isD = T0.Trajectory_History == "Drops";
    fprintf('Rank-sum Rises vs Drops, pressure residual: p = %.3g\n', ...
            ranksum(rP(isR), rP(isD)));
end

OOF = T0(:, {'Source_File','WIRE_uid','Section_mm2','Length_m','Trajectory_History','Flow_Nm3h','Pressure_Pa'});
OOF.LOSO_Pred_Flow_GPR  = Res.LOSO.Q_GP;
OOF.LOSO_Pred_Press_GPR = Res.LOSO.P_GP;
OOF.LOUO_Pred_Flow_GPR  = Res.LOUO.Q_GP;
OOF.LOUO_Pred_Press_GPR = Res.LOUO.P_GP;
OOF.LOUO_Phys_Flow_cal  = Res.LOUO.Q_cal;
OOF.LOUO_Phys_Press_cal = Res.LOUO.P_cal;
OOF.LOLO_Pred_Flow_GPR  = Res.LOLO.Q_GP;
OOF.LOLO_Pred_Press_GPR = Res.LOLO.P_GP;
OOF.LOLO_Phys_Flow_cal  = Res.LOLO.Q_cal;
OOF.LOLO_Phys_Press_cal = Res.LOLO.P_cal;

writetable(OOF, fullfile(outDir, 'CV_OutOfFold_v8.csv'));
save(fullfile(outDir, 'CV_Results_v8.mat'), 'Res', 'T0', 'OOF');
fprintf('Saved CV_Results_v8.mat and CV_OutOfFold_v8.csv\n');
fprintf('Total time: %.1f s\n', toc(tAll));

%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================
function [Qnm3h, dpDarcy, dynSys, Re] = physBaseline(T, rho, RHO_REF, MU_AIR, cd, cf)
% Geometry and weight only. Drag balance with skin friction, solved per row.
    P_AMB_PA = 101325;                % Pa, atmospheric baseline
    d    = T.Wire_OD_mm/1000;             % m, bare wire outer diameter
    Dh   = T.Hydraulic_Diameter_mm/1000;  % m
    Dt   = T.Tube_D_mm/1000;              % m
    Af   = T.Flow_Area_mm2*1e-6;          % m^2, annulus (free flow area)
    At   = pi/4*Dt.^2;                    % m^2, full tube
    
    % --- FERRULE MAPPING ---
    d_ferrule = zeros(size(d));
    for i = 1:numel(d)
        od = T.Wire_OD_mm(i);
        if abs(od - 2.1) < 0.1
            d_ferrule(i) = 2.9 / 1000;
        elseif abs(od - 2.4) < 0.1
            d_ferrule(i) = 3.5 / 1000;
        elseif abs(od - 2.75) < 0.1
            d_ferrule(i) = 4.4 / 1000;
        elseif abs(od - 3.95) < 0.1
            d_ferrule(i) = 5.4 / 1000;
        elseif abs(od - 4.5) < 0.1
            d_ferrule(i) = 6.5 / 1000;
        else
            d_ferrule(i) = d(i);
        end
    end
    
    Aend = pi/4 * d_ferrule.^2;           % m^2, wire end face (Ferrule OD based)
    L    = T.Length_m;                    % m
    W    = T.Wire_Weight_N;               % N

    % Initial velocity guess using form drag component
    v = sqrt(2*W./(rho*cd.*Aend));
    
    converged = false;
    for it = 1:500
        Re_it = rho*v.*Dh/MU_AIR;
        Cf_it = cf*0.079*Re_it.^(-0.25);
        
        vn    = sqrt(2*W./(rho*(cd.*Aend + Cf_it.*pi.*d.*L)));
        
        if max(abs(vn - v)./vn) < 1e-10
            v = vn; converged = true; break
        end
        v = vn;
    end
    if ~converged
        warning('physBaseline: drag iteration not converged (CD=%.3f, CF=%.3f)', cd, cf);
    end

    Re      = rho*v.*Dh/MU_AIR;
    f       = 0.316*Re.^(-0.25);                    % Blasius
    % Absolute inlet pressure = ambient atmospheric pressure + Darcy friction drop
    dpDarcy = P_AMB_PA + (f.*(L./Dh).*rho.*v.^2/2);  % Pa, absolute inlet pressure
    Qnm3h   = 3600*(rho*Af.*v)/RHO_REF;             % Nm^3/h
    dynSys  = 0.5*rho*(Af.*v./At).^2;               % Pa, tube dynamic head
end

function sse = relFlowSSE(T, rho, RHO_REF, MU_AIR, cd, cf)
    Q   = physBaseline(T, rho, RHO_REF, MU_AIR, cd, cf);
    sse = sum(((T.Flow_Nm3h - Q)./T.Flow_Nm3h).^2);
end

function [cd, cf] = calibrateCDCF(T, rho, RHO_REF, MU_AIR, cdLo, cfLo, cfStart)
    obj = @(p) relFlowSSE(T, rho, RHO_REF, MU_AIR, max(p(1),cdLo), max(p(2),cfLo));
    p   = fminsearch(obj, [1.0 cfStart], optimset('Display','off','TolX',1e-6,'TolFun',1e-10));
    cd  = max(p(1), cdLo);
    cf  = max(p(2), cfLo);
end

function cd = calibrateCDOnly(T, rho, RHO_REF, MU_AIR, cdLo)
    obj = @(p) relFlowSSE(T, rho, RHO_REF, MU_AIR, max(p,cdLo), 0);
    p   = fminsearch(obj, 1.0, optimset('Display','off','TolX',1e-6,'TolFun',1e-10));
    cd  = max(p, cdLo);
end

function v = r2(y, yh)
    v = 1 - sum((y - yh).^2)/sum((y - mean(y)).^2);
end

function row = metricRow(name, y, yh)
    e   = y - yh;
    row = table(string(name), r2(y, yh), sqrt(mean(e.^2)), mean(abs(e)), ...
                100*median(abs(e./y)), ...
                'VariableNames', {'Model','R2','RMSE','MAE','MedAbsPctErr'});
end