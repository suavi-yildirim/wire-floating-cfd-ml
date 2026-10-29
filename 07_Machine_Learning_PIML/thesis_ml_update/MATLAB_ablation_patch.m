%% =========================================================================
%  PATCH for CrossVal_Physics_Calibration_Ablation.m  -- feature ablation
%  (UNTESTED in MATLAB: written without access to MATLAB. The Python replication
%   piml_ablation.py gives the numbers of Table "tab:ml_ablation"; run this patch
%   to obtain the same table from your own pipeline.)
%
%  Four edits, all marked "ABLATION".
%% =========================================================================

% ---- (1) After the definition of fP0 (before "for sc = 1:size(schemes,1)") ----
% ABLATION: geometry-only and geometry+weight predictor sets (no physics columns)
fQ_geom  = {'Section_mm2', 'Ferrule_to_Tube_Ratio', 'Blockage_Ratio', 'Length_over_Dh'};
fQ_geomW = [fQ_geom, {'Wire_Weight_N'}];
fP_geom  = {'Section_mm2', 'Tube_D_mm', 'Length_m', 'Blockage_Ratio', 'Length_over_Dh', 'Ferrule_to_Tube_Ratio'};
fP_geomW = [fP_geom, {'Wire_Weight_N'}];
fP_Qphys = [fP_geom, {'Flow_Phys_Nm3h'}];      % pressure model with the flow baseline only

% ---- (2) In the preallocation block of each scheme (next to r.P_GP = nan(n,1);) ----
% ABLATION
r.Q_GP_geom  = nan(n,1);  r.Q_GP_geomW = nan(n,1);
r.P_GP_geom  = nan(n,1);  r.P_GP_geomW = nan(n,1);  r.P_GP_Qphys = nan(n,1);

% ---- (3) In the fold loop, after "mP_GP = fitrgp(...)" ----
% ABLATION: same GPR settings, only the predictor list changes
gp = @(T, y, cols) fitrgp(T, y, 'PredictorNames', cols, ...
        'KernelFunction', 'ardsquaredexponential', 'Standardize', true);
mQ_g  = gp(Tr, 'Flow_Nm3h',   fQ_geom);
mQ_gW = gp(Tr, 'Flow_Nm3h',   fQ_geomW);
mP_g  = gp(Tr, 'Pressure_Pa', fP_geom);
mP_gW = gp(Tr, 'Pressure_Pa', fP_geomW);
mP_Q  = gp(Tr, 'Pressure_Pa', fP_Qphys);

% ---- (3b) In the "out-of-fold predictions" block, after r.P_GP(teMask) = ... ----
% ABLATION
r.Q_GP_geom(teMask)  = predict(mQ_g,  Te);
r.Q_GP_geomW(teMask) = predict(mQ_gW, Te);
r.P_GP_geom(teMask)  = predict(mP_g,  Te);
r.P_GP_geomW(teMask) = predict(mP_gW, Te);
r.P_GP_Qphys(teMask) = predict(mP_Q,  Te);

% ---- (4) In STAGE 3, extend the tables Cq and Cp of every scheme ----
% ABLATION: append these rows inside the "for sc = ..." loop, before disp(Cq) / disp(Cp)
Cq = [Cq; metricRow('GPR, geometry only',      yQ, r.Q_GP_geom);
          metricRow('GPR, geometry + weight',  yQ, r.Q_GP_geomW)];
Cp = [Cp; metricRow('GPR, geometry only',      yP, r.P_GP_geom);
          metricRow('GPR, geometry + weight',  yP, r.P_GP_geomW);
          metricRow('GPR, geometry + Q_phys',  yP, r.P_GP_Qphys)];

% ---- (5) CORRECTION of the pressure baseline basis -------------------------
% Pressure_Pa is a GAUGE pressure (Build_MAT_Dataset.m, line 184). physBaseline adds
% P_AMB_PA to the Darcy drop, so the standalone metric 'Darcy ... (Abs)' compares an
% absolute with a gauge pressure (R2 = -0.04 is a coincidence of the means).
% Either drop "P_AMB_PA +" in physBaseline (the offset is constant, so the surrogates
% are unaffected) or evaluate the baseline as  metricRow(..., yP, r.P_cal - P_AMB_PA).
% Corrected values (Python): R2 = -1.72, MAPE = 76 %.
