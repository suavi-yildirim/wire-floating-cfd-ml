%% =========================================================================
%  CFD DIAGNOSTIC: Per-Case Ferrule & Skin-Friction Coefficient Decomposition
%  Uses exact CFD tube pressures to calculate volume-averaged density rho_b.
%  Input : D:\FAU\Thesis_Data\Report_Data
%  Output: D:\FAU\Thesis_Data\03_PIML\CFD_Diagnostic
%% =========================================================================
clear; clc;

%% ---------------------------- 1. SETTINGS & PATHS ------------------------
thesisRoot = 'D:\FAU\Thesis_Data';
CASE_ROOT  = fullfile(thesisRoot, 'Report_Data');
outDir     = fullfile(thesisRoot, '03_PIML', 'CFD_Diagnostic');

if ~exist(outDir, 'dir'), mkdir(outDir); end

candidatePaths = {
    fullfile(CASE_ROOT, 'CFD_Experimental_Outcome.csv');
    fullfile(CASE_ROOT, 'CFD_Experimental_Outcome.xlsx');
    fullfile(thesisRoot, 'CFD_Experimental_Outcome.csv');
    fullfile(thesisRoot, 'CFD_Experimental_Outcome.xlsx')
};

expCsvPath = '';
for k = 1:numel(candidatePaths)
    if exist(candidatePaths{k}, 'file')
        expCsvPath = candidatePaths{k};
        break;
    end
end
if isempty(expCsvPath)
    error('Could not find CFD_Experimental_Outcome (.csv or .xlsx) under %s', thesisRoot);
end

fprintf('Case Root       : %s\n', CASE_ROOT);
fprintf('Experimental CSV: %s\n', expCsvPath);
fprintf('Output Directory: %s\n\n', outDir);

HOLDOUT_MM2         = 1.5;          % Excluded from calibration (no data leakage)
NARROW_MM2          = [4.0 6.0];    % WC08 narrow-gap cases
EXCLUDE_NARROW_WC08 = true;
IMB_MAX_PCT         = 10.0;          % Convergence threshold: mass imbalance < 1%
MU_AIR              = 1.7894e-5;    % Pa s
R_AIR               = 287.058;      % J/(kg K)
P_AMB_PA            = 101325;       % Pa
T_REF_K             = 293.15;       % 20 degC (CFD operating temperature)
BLASIUS_C           = 0.079;        % Cf = 0.079 * Re^(-0.25)
OUT_CSV             = fullfile(outDir, 'CFD_Coefficients_per_case.csv');

%% ---------------------------- 2. READ EXPERIMENTAL TABLE -----------------
[~, ~, fileExt] = fileparts(expCsvPath);
if strcmpi(fileExt, '.xlsx')
    E = readtable(expCsvPath, 'VariableNamingRule', 'preserve');
else
    E = readtable(expCsvPath, 'Delimiter', ';', 'DecimalSeparator', ',', ...
                  'VariableNamingRule', 'preserve');
end

vn    = E.Properties.VariableNames;
colOf = @(pat) vn{find(contains(vn, pat, 'IgnoreCase', true), 1)};

% Filter out trailing empty rows if present
validRows = ~cellfun(@isempty, E.ID) & (strlength(string(E.ID)) > 0);
E = E(validRows, :);

ids   = string(E.ID);
nC    = height(E);

%% ---------------------------- 3. PARSE FLUENT REPORTS --------------------
rows  = cell(nC, 1);
nSkip = 0;

for k = 1:nC
    folder = fullfile(CASE_ROOT, ids(k));
    if ~isfolder(folder)
        warning('No report folder for %s, skipped.', ids(k));
        nSkip = nSkip + 1;
        continue
    end
    try
        m  = readFluentReport(folder, 'mass-flow-inlet');
        n  = readFluentReport(folder, 'net-mass-flow');
        F  = readFluentReport(folder, 'wire-drag-force');
    catch err
        warning('%s: %s', ids(k), err.message);
        nSkip = nSkip + 1;
        continue
    end

    % Geometry and reference quantities (SI units)
    D       = E.(colOf('Tube_Diameter'))(k) * 1e-3;   % [m]
    dw      = E.(colOf('Wire_OD'))(k)       * 1e-3;   % [m]
    df      = E.(colOf('Ferrule_OD'))(k)    * 1e-3;   % [m]
    L       = E.(colOf('Wire_Length'))(k);            % [m]
    W       = E.(colOf('Wire_Weight'))(k);            % [N]
    sec     = E.(colOf('Cross'))(k);                  % [mm^2]
    tube    = E.(colOf('Tube_Diameter'))(k);          % [mm]
    mdotExp = E.(colOf('Mass_Flow_Target'))(k);       % [kg/s]
    p_raw   = E.(colOf('Pressure'))(k);               % Pressure

    % Convert experimental pressure to Pa
    if p_raw < 10
        p_exp = p_raw * 1e5;      % bar -> Pa
    elseif p_raw < 5000
        p_exp = p_raw * 100;      % mbar -> Pa
    else
        p_exp = p_raw;            % Pa
    end

    % Final iteration values from Fluent reports
    mdot = m.data(end, 2);                            % [kg/s]
    imb  = 100 * n.data(end, 2) / mdot;               % [%]

    % Calculate exact volume-average density in the tube from CFD reports
    try
        % Direct density report if present
        rh = readFluentReport(folder, 'rho-tube-vol');
        rb = rh.data(end, 2);
    catch
        % Fallback: Calculate from start-tube and silencer pressure reports
        try
            p_in_rep  = readFluentReport(folder, 'start-tube-outlet-pressure');
            p_out_rep = readFluentReport(folder, 'silencer-inlet-pressure');
            p_tube_gauge = 0.5 * (p_in_rep.data(end, 2) + p_out_rep.data(end, 2));
        catch
            try
                dp_rep    = readFluentReport(folder, 'tube-pressure-drop');
                p_out_rep = readFluentReport(folder, 'silencer-inlet-pressure');
                p_tube_gauge = p_out_rep.data(end, 2) + 0.5 * dp_rep.data(end, 2);
            catch
                % Fallback if only experimental pressure is known
                p_tube_gauge = 0.5 * p_exp;
            end
        end
        p_mean_abs = P_AMB_PA + p_tube_gauge;
        rb         = p_mean_abs / (R_AIR * T_REF_K);
    end

    iB = find(contains(F.names, 'bottom'), 1);
    iT = find(contains(F.names, 'top'), 1);
    iW = find(contains(F.names, '(wall_wire)'), 1);
    if isempty(iB) || isempty(iT) || isempty(iW)
        warning('%s: force column missing, skipped.', ids(k));
        nSkip = nSkip + 1;
        continue
    end
    Fb = F.data(end, iB);   
    Ft = F.data(end, iT);   
    Fw = F.data(end, iW);

    % Aerodynamic definitions
    Dh      = D - dw;                                 % [m]
    A_ann   = pi * (D^2 - dw^2) / 4;                  % [m^2]
    A_ferr  = pi * df^2 / 4;                          % [m^2]
    A_wire  = pi * dw^2 / 4;                          % [m^2]
    vb      = mdot / (rb * A_ann);                    % [m/s]
    q_b     = 0.5 * rb * vb^2;                        % [Pa]
    Re      = rb * vb * Dh / MU_AIR;                  % [-]

    Cf      = Fw / (q_b * pi * dw * L);
    CfRatio = Cf / (BLASIUS_C * Re^(-0.25));
    f_darcy = 4 * BLASIUS_C * Re^(-0.25);             % Darcy f = 4 * Cf

    CD_bot  = Fb / (q_b * A_ferr);
    CD_top  = Ft / (q_b * A_ferr);
    CDend   = (Fb + Ft) / (q_b * A_ferr);

    rows{k} = struct('ID', ids(k), 'Section_mm2', sec, 'Tube_mm', tube, ...
                     'Length_m', L, 'Wire_Weight_N', W, ...
                     'Imbalance_pct', imb, 'mdot_CFD', mdot, 'mdot_exp', mdotExp, ...
                     'F_bottom', Fb, 'F_top', Ft, 'F_wall', Fw, ...
                     'rho_b', rb, 'v_b', vb, 'q_b', q_b, 'Re_Dh', Re, 'Dh_m', Dh, ...
                     'Cf', Cf, 'CfRatio', CfRatio, 'f_darcy', f_darcy, ...
                     'CD_bot', CD_bot, 'CD_top', CD_top, 'CDend', CDend, ...
                     'A_wire_over_A_ferr', A_wire / A_ferr, 'p_exp_Pa', p_exp);
end

rows = rows(~cellfun(@isempty, rows));
T    = struct2table([rows{:}]');

%% ---------------------------- 4. FILTERING & DIAGNOSTIC CHECKS -----------
isHold   = abs(T.Section_mm2 - HOLDOUT_MM2) < 1e-6;
isNarrow = T.Tube_mm == 8 & ismember(round(T.Section_mm2, 1), NARROW_MM2);
isConv   = abs(T.Imbalance_pct) < IMB_MAX_PCT;
use      = ~isHold & isConv & ~(EXCLUDE_NARROW_WC08 & isNarrow);
T.Used   = use;

fprintf('\nCases read: %d (skipped: %d) | Used for calibration: %d\n', ...
        height(T), nSkip, nnz(use));

S = T(use, :);
S.dP_over_q   = S.f_darcy .* S.Length_m ./ S.Dh_m;                   
S.dp_Pa       = S.dP_over_q .* S.q_b;                                
S.dp_ratio    = S.dp_Pa ./ S.p_exp_Pa;                               
S.CD0_full    = S.CDend - S.dP_over_q;                               
S.CD0_wire    = S.CDend - S.dP_over_q .* S.A_wire_over_A_ferr;       

disp(S(:, {'ID','Tube_mm','Section_mm2','Length_m','CfRatio','CD_bot','CD_top','CDend','dp_Pa','dp_ratio','CD0_wire'}));

fprintf('\n--- SUMMARY PER TUBE ---\n');
for it = unique(S.Tube_mm)'
    g = S.Tube_mm == it;
    fprintf('WC%02d (n=%2d): CF_RATIO med = %.3f | CDend med = %.3f [%.2f .. %.2f] | CD0_wire med = %.3f [%.2f .. %.2f]\n', ...
            it, nnz(g), median(S.CfRatio(g)), ...
            median(S.CDend(g)), min(S.CDend(g)), max(S.CDend(g)), ...
            median(S.CD0_wire(g)), min(S.CD0_wire(g)), max(S.CD0_wire(g)));
end

writetable(T, OUT_CSV);
fprintf('\nSaved diagnostic CSV to: %s\n', OUT_CSV);

%% =========================================================================
%  LOCAL FUNCTION
%% =========================================================================
function rep = readFluentReport(folder, prefix)
    f = dir(fullfile(folder, [prefix '*.out']));
    if isempty(f)
        error('No report "%s*.out" in %s', prefix, folder);
    end
    txt = splitlines(string(fileread(fullfile(f(1).folder, f(1).name))));
    txt = txt(strlength(strtrim(txt)) > 0);
    tk  = regexp(char(txt(3)), '"([^"]+)"', 'tokens');
    rep.names = cellfun(@(t) t{1}, tk, 'UniformOutput', false);
    rows = txt(4:end);
    rep.data = zeros(numel(rows), numel(rep.names));
    for i = 1:numel(rows)
        rep.data(i, :) = sscanf(char(rows(i)), '%f')';
    end
end