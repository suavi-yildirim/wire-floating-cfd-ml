%% =========================================================================
%  WFE DATA PIPELINE v8: raw WFE logs -> physics-informed ML dataset
%
%  Changes vs v5 (after review):
%   1. Diagnostic density bounded: ambient pressure (lower bound at the wire)
%      and inlet gauge pressure (upper bound). Both Darcy diagnostics reported.
%   2. Flow baseline: drag balance includes wire skin friction,
%        W = 1/2 rho v^2 (C_D A_end + C_f pi d L), solved per row.
%      Dimensional check: m_dot = rho A v = A sqrt(2 W rho /(C_D A_end + ...)),
%      and Q_std = 3600 m_dot / rho_ref is dimensionally exact.
%   3. Pressure baseline: Darcy-Weisbach annular drop at the drag-balance
%      velocity (geometry + weight only). W/A_end kept only as Diag_Piston_Pa.
%   4. Trajectory_History stays out of the baselines (analyse hysteresis in residuals).
%   5. Progress printed to the Command Window per file, per stage, with timing.
%
%  Regression Learner (one session per response):
%    PRESSURE: response Pressure_Pa ; predictors Section_mm2, Tube_D_mm, Length_m,
%              Blockage_Ratio, Length_over_Dh, Pressure_Phys_Darcy_Pa
%    FLOW:     response Flow_Nm3h   ; predictors Section_mm2, Tube_D_mm, Length_m,
%              Blockage_Ratio, Length_over_Dh, Flow_Phys_Nm3h
%
%  Never predictors: Diag_*, Flow_kgs, Temp_*, Air_Density_amb_kgm3,
%                    Trajectory_History, Stable_*, Sensor_*, Phases_Used, *_RelSpread
%% =========================================================================
clc; clear; close all;
tAll = tic;

%% ============================ SETTINGS ===================================
inputDir   = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), '01_WFE');
outputDir  = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'MAT_Dataset_v8');
mlTableDir = fullfile(outputDir, 'ML_Ready_Tables');

FLOW_RAW_TO_NM3H = 1/100;     % raw flow -> Nm^3/h (inherited, VERIFY)
TEMP_RAW_TO_C    = 1/10;      % raw temperature -> degC (inherited, VERIFY)
MBAR_TO_PA       = 100;

R_AIR    = 287.058;           % J/(kg K), dry air (reproduces workbook Air_Density)
P_AMB_PA = 101325;            % Pa, ambient pressure (reproduces workbook Air_Density)
T_REF_K  = 273.15;            % flow sensor reference temperature, 0 degC
RHO_REF  = P_AMB_PA / (R_AIR * T_REF_K);   % 1.292248 kg/m^3
MU_AIR   = 1.7894e-5;         % Pa s, dynamic viscosity of air (CFD setup)

CD       = 1.168;               % form drag coefficient (CFD median result)
CF_RATIO = 1.145;               % wire skin friction relative to 0.079 Re^-0.25 (CFD: 1.12-1.32)

MIN_PHASE_MS           = 1000;      % thesis tau_min
EXCLUDE_SECTION_MM2    = 2.5;       % excluded; document the reason in the thesis
ALLOWED_STABLE_SENSORS = [18 19 20];% process_data.py validation rule (diagnostic)

% UID range -> [UID_lo  UID_hi  Section_mm2  Insulated_OD_mm]
UID_MAP = [ 1  4   0.5  2.10;
            5  8   1.0  2.40;
            9 12   1.5  2.75;
           17 20   4.0  3.95;
           21 24   6.0  4.50];

% UID -> [UID  Length_m  Wire_Weight_N]   (CFD_Experimental_Outcome.xlsx, Wire_Weight[N])
WEIGHT_TABLE = [ ...
    1   0.5  0.041202;
    2   1.0  0.082404;
    3   2.0  0.164808;
    4   3.5  0.328635;
    5   0.5  0.064746;
    6   1.0  0.129492;
    7   2.0  0.258003;
    8   3.5  0.514044;
    9   0.5  0.093195;
   10   1.0  0.184428;
   11   2.0  0.369837;
   12   3.5  0.737712;
   17   0.5  0.218763;
   18   1.0  0.437526;
   19   2.0  0.873090;
   20   3.5  1.743237;
   21   0.5  0.309996;
   22   1.0  0.616068;
   23   2.0  1.226250;
   24   3.5  2.460348];

if ~exist(mlTableDir, 'dir'), mkdir(mlTableDir); end

%% ============================ STAGE 1: SCAN ==============================
fprintf('=== WFE pipeline v8 ===\n');
fprintf('Stage 1/4: scanning %s\n', inputDir);
files  = dir(fullfile(inputDir, 'WFE_*.csv'));       % raw logs only
nFiles = numel(files);
fprintf('  %d raw WFE log files found (summary CSVs are ignored).\n', nFiles);
fprintf('  Reference air density (0 degC): %.6f kg/m^3\n', RHO_REF);

%% ============================ STAGE 2: PARSE =============================
fprintf('Stage 2/4: parsing logs and extracting stable runs\n');
rowsCell = {};
logs     = strings(0, 3);      % File | Status | Reason
tParse   = tic;

for i = 1:nFiles
    fname = files(i).name;
    txt   = readTextFile(fullfile(inputDir, fname));

    % ---- 1) Header metadata and controller stability -------------------
    meta = parseMetadata(txt);
    if isempty(meta)
        reason = "metadata missing or unreadable";
        logs(end+1,:) = [string(fname) "SKIPPED" reason];
        printStatus(i, nFiles, fname, "SKIPPED", reason);
        continue
    end
    if ~meta.stable
        reason = "controller reports not stable";
        logs(end+1,:) = [string(fname) "SKIPPED" reason];
        printStatus(i, nFiles, fname, "SKIPPED", reason);
        continue
    end

    % ---- 2) Section, OD, length and weight from the UID ----------------
    r = find(meta.uid >= UID_MAP(:,1) & meta.uid <= UID_MAP(:,2), 1);
    if isempty(r)
        reason = sprintf('UID %d not in UID_MAP', meta.uid);
        logs(end+1,:) = [string(fname) "SKIPPED" string(reason)];
        printStatus(i, nFiles, fname, "SKIPPED", reason);
        continue
    end
    wk = find(WEIGHT_TABLE(:,1) == meta.uid, 1);
    if isempty(wk)
        reason = sprintf('UID %d not in WEIGHT_TABLE', meta.uid);
        logs(end+1,:) = [string(fname) "SKIPPED" string(reason)];
        printStatus(i, nFiles, fname, "SKIPPED", reason);
        continue
    end

    warnMsg = "";
    if abs(meta.section - UID_MAP(r,3)) > 0.05
        warnMsg = sprintf('header section %.2f replaced by UID %d value %.1f mm^2', ...
                          meta.section, meta.uid, UID_MAP(r,3));
    end
    if abs(meta.length_m - WEIGHT_TABLE(wk,2)) > 0.05
        warnMsg = sprintf('header length %.2f replaced by UID %d table length %.2f m', ...
                          meta.length_m, meta.uid, WEIGHT_TABLE(wk,2));
    end
    meta.section  = UID_MAP(r,3);
    meta.length_m = WEIGHT_TABLE(wk,2);
    wireOD = UID_MAP(r,4);                 % mm
    W_N    = WEIGHT_TABLE(wk,3);           % N

    % ---- 3) Stable phases and sensor history ---------------------------
    [headerLine, stableLines, sensorLines] = parseBlocks(txt);
    if isempty(stableLines)
        reason = "no stable phase block";
        logs(end+1,:) = [string(fname) "SKIPPED" reason];
        printStatus(i, nFiles, fname, "SKIPPED", reason);
        continue
    end
    T = parseStableTable(headerLine, stableLines);
    needed = ["Phase","Duration","Pressure_Stable_Min","Pressure_Stable_Max", ...
              "Pressure_Stable_Average","Flow_Stable_Min","Flow_Stable_Max", ...
              "Flow_Stable_Average","Temp_Stable_Average"];
    if isempty(T) || ~all(ismember(needed, string(T.Properties.VariableNames)))
        reason = "stable table unreadable or columns missing";
        logs(end+1,:) = [string(fname) "SKIPPED" reason];
        printStatus(i, nFiles, fname, "SKIPPED", reason);
        continue
    end
    if height(T) ~= numel(stableLines)
        reason = "stable table row count mismatch";
        logs(end+1,:) = [string(fname) "SKIPPED" reason];
        printStatus(i, nFiles, fname, "SKIPPED", reason);
        continue
    end
    rawP = cellfun(@(s) strsplit(s, ';'), stableLines, 'UniformOutput', false);

    % ---- 4) Phases >= tau_min, duration-weighted run values ------------
    ok = T.Duration >= MIN_PHASE_MS;
    if ~any(ok)
        reason = "no stable phase >= 1 s";
        logs(end+1,:) = [string(fname) "SKIPPED" reason];
        printStatus(i, nFiles, fname, "SKIPPED", reason);
        continue
    end
    Tk   = T(ok,:);
    kIdx = find(ok);
    w    = Tk.Duration;
    wSum = sum(w);

    p_avg = sum(Tk.Pressure_Stable_Average .* w) / wSum * MBAR_TO_PA;       % Pa (gauge)
    q_avg = sum(Tk.Flow_Stable_Average    .* w) / wSum * FLOW_RAW_TO_NM3H;  % Nm^3/h
    T_C   = sum(Tk.Temp_Stable_Average    .* w) / wSum * TEMP_RAW_TO_C;     % degC
    p_min = min(Tk.Pressure_Stable_Min) * MBAR_TO_PA;
    p_max = max(Tk.Pressure_Stable_Max) * MBAR_TO_PA;
    q_min = min(Tk.Flow_Stable_Min)     * FLOW_RAW_TO_NM3H;
    q_max = max(Tk.Flow_Stable_Max)     * FLOW_RAW_TO_NM3H;

    if ~(p_avg > 0 && q_avg > 0)
        reason = "non-positive mean pressure or flow";
        logs(end+1,:) = [string(fname) "SKIPPED" reason];
        printStatus(i, nFiles, fname, "SKIPPED", reason);
        continue
    end
    p_rel = (p_max - p_min) / p_avg;     % diagnostic
    q_rel = (q_max - q_min) / q_avg;     % diagnostic

    % ---- 5) Approach direction (thesis rule, representative phase) -----
    repIdx = kIdx(end);
    [traj, stSensor, sensValid] = classifyMotion(sensorLines, ...
        strtrim(rawP{repIdx}{2}), strtrim(rawP{repIdx}{3}), T.Duration(repIdx), ...
        ALLOWED_STABLE_SENSORS);

    % ---- 6) Geometry ---------------------------------------------------
    D            = meta.tube_mm;                 % mm
    d_mm         = wireOD;                       % mm
    L            = meta.length_m;                % m
    flowArea_mm2 = pi/4 * (D^2 - d_mm^2);        % mm^2
    Dh_mm        = D - d_mm;                     % mm
    cat          = sprintf('WC%02d_%.1fmm', round(D), meta.section);
    cfg          = sprintf('%s_L%.2f', cat, L);

    % ---- 7) Diagnostics from measured flow (NOT predictors) ------------
    d_m    = d_mm / 1000;
   
    % Ferrule mapping for accurate end-face area calculation
    d_ferrule_mm = getFerruleOD(d_mm);
    d_ferrule_m  = d_ferrule_mm / 1000;
    A_end  = pi/4 * d_ferrule_m^2;               % m^2, ferrule end face
    
    A_flow = flowArea_mm2 * 1e-6;                % m^2
    Dh_m   = Dh_mm / 1000;                       % m

    T_K     = T_C + 273.15;                      % K
    rho_amb = P_AMB_PA / (R_AIR * T_K);          % kg/m^3, lower bound of density at wire
    rho_in  = (P_AMB_PA + p_avg) / (R_AIR * T_K);% kg/m^3, upper bound (inlet gauge pressure)
    mdot    = (q_avg / 3600) * RHO_REF;          % kg/s, from standard flow
    v_amb   = mdot / (rho_amb * A_flow);         % m/s, upper bound of velocity
    v_in    = mdot / (rho_in  * A_flow);         % m/s, lower bound of velocity
    Re_Dh   = mdot * Dh_m / (A_flow * MU_AIR);   % density-independent
    f_bl    = 0.316 * Re_Dh^(-0.25);             % Blasius, valid for Re < 1e5
    dp_amb  = f_bl * (L / Dh_m) * rho_amb * v_amb^2 / 2;   % Pa, upper bound
    dp_in   = f_bl * (L / Dh_m) * rho_in  * v_in^2  / 2;   % Pa, lower bound
    piston  = W_N / A_end;                       % Pa, end-face piston balance (diagnostic)

    % ---- 8) One row per accepted run -----------------------------------
    s = struct();
    s.Source_File           = string(fname);
    s.WIRE_uid              = meta.uid;
    s.Category              = string(cat);
    s.Config_ID             = string(cfg);
    s.Date                  = string(meta.date);
    s.Section_mm2           = meta.section;
    s.Wire_OD_mm            = d_mm;

    % --- ADDING FERRULE PREDICTORS FOR MACHINE LEARNING ---
    s.Ferrule_OD_mm         = d_ferrule_mm;
    s.Ferrule_Area_mm2      = A_end * 1e6;       % Converted back to mm^2 for clean ML scaling
    s.Ferrule_to_Tube_Ratio = (A_end * 1e6) / (pi/4 * meta.tube_mm^2);

    s.Tube_D_mm             = D;
    s.Length_m              = L;
    s.Flow_Area_mm2         = flowArea_mm2;
    s.Hydraulic_Diameter_mm = Dh_mm;
    s.Blockage_Ratio        = d_mm / D;
    s.Length_over_Dh        = (L*1000) / Dh_mm;
    s.Wire_Weight_N         = W_N;
    s.Pressure_Pa           = p_avg;
    s.Flow_Nm3h             = q_avg;
    s.Flow_kgs              = mdot;
    s.Temp_C                = T_C;
    s.Temp_K                = T_K;
    s.Air_Density_amb_kgm3  = rho_amb;
    s.Diag_Piston_Pa        = piston;
    s.Diag_Velocity_amb_ms  = v_amb;
    s.Diag_Re_Dh            = Re_Dh;
    s.Diag_DP_Darcy_amb_Pa  = dp_amb;
    s.Diag_DP_Darcy_inlet_Pa= dp_in;
    s.Trajectory_History    = string(traj);
    s.Stable_Sensor_ID      = stSensor;
    s.Sensor_Set_Valid      = sensValid;
    s.Phases_Used           = height(Tk);
    s.Stable_Duration_ms    = wSum;
    s.Pressure_RelSpread    = p_rel;
    s.Flow_RelSpread        = q_rel;
    s.Stable_OK             = true;

    rowsCell{end+1,1} = struct2table(s); %#ok<SAGROW>

    if warnMsg ~= ""
        logs(end+1,:) = [string(fname) "WARN" string(warnMsg)];
        printStatus(i, nFiles, fname, "OK+WARN", warnMsg);
    else
        logs(end+1,:) = [string(fname) "OK" ""];
        printStatus(i, nFiles, fname, "OK", "");
    end
end
fprintf('  Stage 2 done in %.1f s: %d runs accepted, %d files skipped.\n', ...
        toc(tParse), nnz(strcmp(logs(:,2),"OK") | strcmp(logs(:,2),"OK+WARN") | logs(:,2)=="WARN"), ...
        nnz(logs(:,2) == "SKIPPED"));

if isempty(rowsCell)
    error('No usable runs. Check %s.', fullfile(outputDir, 'build_log.csv'));
end
ML_Data_All = vertcat(rowsCell{:});

%% ============================ STAGE 3: PHYSICS ===========================
%% ============================ STAGE 3: PHYSICS ===========================
fprintf('Stage 3/4: physics baselines (drag balance with skin friction, Darcy pressure)\n');
ML_Data_All = ML_Data_All(abs(ML_Data_All.Section_mm2 - EXCLUDE_SECTION_MM2) > 1e-6, :);
fprintf('  Runs after section exclusion: %d\n', height(ML_Data_All));

% Fixed operating density (mean measured temperature): baselines use geometry and weight only
T_OP_K = mean(ML_Data_All.Temp_K);
rho_op = P_AMB_PA / (R_AIR * T_OP_K);
fprintf('  Operating density: T = %.2f K, rho = %.4f kg/m^3\n', T_OP_K, rho_op);

d_m  = ML_Data_All.Wire_OD_mm / 1000;
Dh_m = ML_Data_All.Hydraulic_Diameter_mm / 1000;

% Ferrule mapping for dataset-wide physics calculation
d_ferrule_m = arrayfun(@(x) getFerruleOD(x)/1000, ML_Data_All.Wire_OD_mm);
A_end  = pi/4 * d_ferrule_m.^2;          % Ferrule OD based end-face area

A_flow = ML_Data_All.Flow_Area_mm2 * 1e-6;
L      = ML_Data_All.Length_m;
W      = ML_Data_All.Wire_Weight_N;

% Drag balance with skin friction, solved per row by fixed-point iteration
v = sqrt(2 .* W ./ (rho_op * CD .* A_end));                 % initial guess (ferrule based)
converged = false;
for it = 1:500
    Re_it = rho_op .* v .* Dh_m ./ MU_AIR;
    Cf_it = CF_RATIO .* 0.079 .* Re_it .^ (-0.25);
    v_new = sqrt(2 .* W ./ (rho_op .* (CD .* A_end + Cf_it .* pi .* d_m .* L)));
    if max(abs(v_new - v) ./ v_new) < 1e-10
        v = v_new;  converged = true;  break
    end
    v = v_new;
end
if converged
    fprintf('  Drag-balance iteration converged in %d steps.\n', it);
else
    warning('Drag-balance iteration did not converge; check inputs.');
end

Re_op = rho_op .* v .* Dh_m ./ MU_AIR;
f_op  = 0.316 .* Re_op .^ (-0.25);                          % Blasius, valid Re < 1e5

% Add ambient pressure (101325 Pa) to the Darcy friction drop
ML_Data_All.Pressure_Phys_Darcy_Pa = P_AMB_PA + (f_op .* (L ./ Dh_m) .* rho_op .* v.^2 ./ 2);
ML_Data_All.Flow_Phys_Nm3h         = 3600 .* (rho_op .* A_flow .* v) ./ RHO_REF;   % Nm^3/h

fprintf('  Reynolds number range (operating point): %.0f - %.0f (Blasius valid < 1e5)\n', ...
        min(Re_op), max(Re_op));

%% ============================ STAGE 4: SAVE ==============================
fprintf('Stage 4/4: saving outputs to %s\n', outputDir);
save(fullfile(mlTableDir, 'ML_Data_All_v8.mat'), 'ML_Data_All');
writetable(ML_Data_All, fullfile(outputDir, 'ML_Data_All_v8.csv'));

cats = unique(ML_Data_All.Category);
for k = 1:numel(cats)
    ML_Data = ML_Data_All(ML_Data_All.Category == cats(k), :);
    save(fullfile(mlTableDir, sprintf('ML_Table_%s.mat', cats(k))), 'ML_Data');
end
writetable(array2table(logs, 'VariableNames', {'File','Status','Reason'}), ...
           fullfile(outputDir, 'build_log.csv'));

%% ============================ SUMMARY / CHECKS ===========================
fprintf('\n=== SUMMARY ===\n');
fprintf('Runs accepted (after 2.5 mm exclusion): %d   (thesis: 793)\n', height(ML_Data_All));
fprintf('Skipped files: %d   Warnings: %d\n', ...
        nnz(logs(:,2) == "SKIPPED"), nnz(logs(:,2) == "WARN"));
disp(groupcounts(ML_Data_All, 'Tube_D_mm'));
fprintf('Approach direction (thesis: Rises 445, Drops 279, Unknown 69):\n');
disp(groupcounts(ML_Data_All, 'Trajectory_History'));
fprintf('Physics checks, median measured / baseline:\n');
fprintf('  Pressure vs Darcy at drag velocity:  %.2f\n', ...
        median(ML_Data_All.Pressure_Pa ./ ML_Data_All.Pressure_Phys_Darcy_Pa));
fprintf('  Flow vs drag balance with friction:  %.2f\n', ...
        median(ML_Data_All.Flow_Nm3h ./ ML_Data_All.Flow_Phys_Nm3h));
fprintf('  Diagnostic Darcy (ambient rho):      %.2f\n', ...
        median(ML_Data_All.Pressure_Pa ./ ML_Data_All.Diag_DP_Darcy_amb_Pa));
fprintf('  Diagnostic Darcy (inlet rho):        %.2f\n', ...
        median(ML_Data_All.Pressure_Pa ./ ML_Data_All.Diag_DP_Darcy_inlet_Pa));

fprintf('\nREGRESSION LEARNER - PRESSURE session\n');
fprintf('  Response:   Pressure_Pa\n');
fprintf('  Predictors: Section_mm2, Tube_D_mm, Length_m, Blockage_Ratio, Length_over_Dh, Pressure_Phys_Darcy_Pa\n');
fprintf('\nREGRESSION LEARNER - FLOW session\n');
fprintf('  Response:   Flow_Nm3h\n');
fprintf('  Predictors: Section_mm2, Tube_D_mm, Length_m, Blockage_Ratio, Length_over_Dh, Flow_Phys_Nm3h\n');
fprintf('\nTotal time: %.1f s\n', toc(tAll));
fprintf('Build log: %s\n', fullfile(outputDir, 'build_log.csv'));

%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================
function printStatus(i, n, fname, status, msg)
    fprintf('[%4d/%4d] %-34s %-9s %s\n', i, n, fname, status, msg);
end

function txt = readTextFile(p)
    fid = fopen(p, 'rt');
    if fid < 0, error('Cannot open %s', p); end
    c   = onCleanup(@() fclose(fid));  %#ok<NASGU>
    txt = fread(fid, '*char')';
end

function meta = parseMetadata(txt)
    meta = [];
    tok  = @(pat) regexp(txt, pat, 'tokens', 'once');
    u  = tok('WIRE_uid:([\d,]+);');
    s  = tok('WIRE_section:([\d,]+);');
    l  = tok('WIRE_lenght:([\d,]+);');      % raw-file spelling "lenght" is intentional
    t  = tok('TUBE_diameter:([\d,]+);');
    dt = tok('Date:(.*?)\r?\n');
    sr = tok('stable:\s*(True|False)');
    if any(cellfun(@isempty, {u, s, l, t, dt, sr})), return; end
    num = @(x) str2double(strrep(x{1}, ',', '.'));
    meta.uid      = num(u);
    meta.section  = num(s);
    meta.length_m = num(l);
    meta.tube_mm  = num(t);
    meta.date     = strtrim(dt{1});
    meta.stable   = strcmpi(strtrim(sr{1}), 'True');
    if any(isnan([meta.uid meta.section meta.length_m meta.tube_mm]))
        meta = [];
    end
end

function [headerLine, stableLines, sensorLines] = parseBlocks(txt)
    lines = splitlines(txt);
    headerLine  = '';
    stableLines = {};
    sensorLines = {};
    mode = 'none';
    for j = 1:numel(lines)
        line = strtrim(lines{j});
        if startsWith(line, 'STABLE PHASES')
            mode = 'stable'; continue
        elseif startsWith(line, 'ALL SENSOR CHANGES')
            mode = 'sensor'; continue
        elseif startsWith(line, 'INCREASES') || startsWith(line, 'DECREASES') || ...
               startsWith(line, 'SYSTEM FLUID RESPONSE')
            mode = 'none'; continue
        end
        if isempty(line), continue; end
        switch mode
            case 'stable'
                if startsWith(line, 'Phase;Start_Time')
                    headerLine = line;
                else
                    stableLines{end+1,1} = line; %#ok<AGROW>
                end
            case 'sensor'
                if ~startsWith(line, 'S;Changed_Time')
                    sensorLines{end+1,1} = line; %#ok<AGROW>
                end
        end
    end
end

function T = parseStableTable(headerLine, stableLines)
    T = table();
    if isempty(headerLine), return; end
    tmp = [tempname '.csv'];
    c   = onCleanup(@() cleanupFile(tmp));  %#ok<NASGU>
    fid = fopen(tmp, 'w');
    fprintf(fid, '%s\n', headerLine);
    fprintf(fid, '%s\n', stableLines{:});
    fclose(fid);
    T = readtable(tmp, 'Delimiter', ';', 'DecimalSeparator', ',', 'TextType', 'char');
end

function cleanupFile(f)
    if exist(f, 'file'), delete(f); end
end

function t = parseDT(s)
    try
        t = datetime(strtrim(s), 'InputFormat', 'dd.MM.yyyy HH:mm:ss,SSS');
    catch
        t = NaT;
    end
end

function ev = parseSensorEvents(sensorLines)
    ev = struct('sensor', {}, 't', {}, 'state', {});
    for k = 1:numel(sensorLines)
        p = strsplit(sensorLines{k}, ';');
        if numel(p) < 3, continue; end
        lab = strtrim(p{1});
        if ~startsWith(lab, 'S'), continue; end
        sn = str2double(lab(2:end));
        if isnan(sn), continue; end
        res = strtrim(p{3});
        if ~any(strcmpi(res, {'True','False'})), continue; end
        t = parseDT(p{2});
        if isnat(t), continue; end
        ev(end+1) = struct('sensor', sn, 't', t, 'state', strcmpi(res, 'True')); %#ok<AGROW>
    end
    if ~isempty(ev)
        [~, ord] = sort([ev.t]);
        ev = ev(ord);
    end
end

function [traj, stableSensor, sensorValid] = classifyMotion(sensorLines, startStr, endStr, durMs, allowed)
    % Port of process_data.py :: classify_motion_from_sensors
    traj = 'Unknown';  stableSensor = NaN;  sensorValid = false;

    ev = parseSensorEvents(sensorLines);
    if isempty(ev), return; end

    t0 = parseDT(startStr);
    if isnat(t0), return; end
    t1 = parseDT(endStr);
    if isnat(t1) || year(t1) < 2000
        t1 = t0 + milliseconds(durMs);
    end

    sIDs = unique([ev.sensor]);
    st   = true(size(sIDs));
    for k = 1:numel(ev)
        if ev(k).t <= t0
            st(sIDs == ev(k).sensor) = ev(k).state;
        end
    end
    activeSet = sIDs(st);

    for k = 1:numel(ev)
        if ev(k).t >= t0 && ev(k).t <= t1 && ev(k).state
            activeSet = union(activeSet, ev(k).sensor);
        end
    end
    if isempty(activeSet), return; end

    stableSensor = min(activeSet);
    sensorValid  = all(ismember(activeSet, allowed));

    prevSensor = NaN;
    before = find([ev.t] < t0);
    for k = fliplr(before)
        if ev(k).sensor ~= stableSensor
            prevSensor = ev(k).sensor;
            break
        end
    end
    if isnan(prevSensor), return; end

    if prevSensor < stableSensor
        traj = 'Rises';
    elseif prevSensor > stableSensor
        traj = 'Drops';
    else
        traj = 'Unknown';
    end
end

function od_f = getFerruleOD(od_wire)
% Map wire outer diameter (mm) to its corresponding Ferrule outer diameter (mm)
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
    od_f = od_wire; % Fallback to wire OD if no match found
end
end