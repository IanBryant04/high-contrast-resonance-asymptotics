% Standalone, answer-agnostic wavelength diagnostic for the current
% silica-core/gold-shell geometry. No COMSOL result is used by the solve.
clc
clearvars
close all

%% Forward inputs
r_core_nm = 71.9060;
t_shell_nm = 8.2508;
r_outer_nm = r_core_nm + t_shell_nm;
rho = r_core_nm/r_outer_nm;

n_medium = 1.0;
n_silica = 1.45;
eta_core = n_silica^2/n_medium^2 - 1;

% This broad interval identifies the application-relevant NIR branch. It
% does not tell the solver to select the mode closest to 800 nm.
material_scan_nm = (600:10:1200).';
branch_window_nm = [500 1600];
N = 180;

% Used only after prediction to calculate a validation error. Set to [] to
% suppress the comparison; it never enters the material or mode selection.
comparison_peak_nm = 800;

%% Build the limiting radial integral operator once
r = linspace(1e-12,1,N).';
dr = r(2)-r(1);
w = ones(N,1)*dr;
w([1 end]) = dr/2;

kernel = zeros(N,N);
for i = 1:N
    ri = r(i);
    inside = r<=ri;
    outside = ~inside;
    kernel(i,inside) = (r(inside).^2.*w(inside)/ri).';
    kernel(i,outside) = (r(outside).*w(outside)).';
end

%% Coarse dispersive fixed-point scan
n_scan = numel(material_scan_nm);
predicted_nm = nan(n_scan,1);
brightness = nan(n_scan,1);
lambda0_scan = nan(n_scan,1)+1i*nan(n_scan,1);
eigen_residual = nan(n_scan,1);

fprintf('\nDISPERSIVE FRAMEWORK WAVELENGTH DIAGNOSTIC\n');
fprintf('Geometry: core %.4f nm, shell %.4f nm, outer %.4f nm\n', ...
    r_core_nm,t_shell_nm,r_outer_nm);
fprintf('Material scan: %.0f to %.0f nm; NIR branch: %.0f to %.0f nm\n\n', ...
    material_scan_nm(1),material_scan_nm(end), ...
    branch_window_nm(1),branch_window_nm(2));

for j = 1:n_scan
    [predicted_nm(j),lambda0_scan(j),brightness(j),eigen_residual(j)] = ...
        predict_at_material_wavelength(material_scan_nm(j),kernel,r,w,rho, ...
        eta_core,n_medium,r_outer_nm,t_shell_nm,branch_window_nm);
end

fixed_point_residual_nm = predicted_nm-material_scan_nm;
valid = isfinite(fixed_point_residual_nm);

% Locate every sign change before refining. This avoids supplying an 800 nm
% answer as the starting point or asking for the mode nearest that answer.
brackets = zeros(0,2);
for j = 1:n_scan-1
    if valid(j) && valid(j+1)
        if fixed_point_residual_nm(j)==0
            brackets(end+1,:) = [material_scan_nm(j) material_scan_nm(j)]; %#ok<AGROW>
        elseif sign(fixed_point_residual_nm(j))~=sign(fixed_point_residual_nm(j+1))
            brackets(end+1,:) = material_scan_nm(j:j+1).'; %#ok<AGROW>
        end
    end
end

if isempty(brackets)
    fprintf('FAIL: no self-consistent wavelength was bracketed.\n');
    if any(valid)
        [minimum_mismatch_nm,index] = min(abs(fixed_point_residual_nm(valid)));
        valid_indices = find(valid);
        index = valid_indices(index);
        fprintf('Smallest mismatch = %.3f nm at material wavelength %.3f nm.\n', ...
            minimum_mismatch_nm,material_scan_nm(index));
    else
        fprintf('No admissible mode was found anywhere in the NIR window.\n');
    end
    fprintf(['The old conversion does not provide a forward wavelength for ', ...
        'this branch over the stated scan.\n']);
    roots_nm = [];
    root_lambda0 = [];
    root_brightness = [];
    root_eigen_residual = [];
else
    roots_nm = nan(size(brackets,1),1);
    root_lambda0 = nan(size(brackets,1),1)+1i*nan(size(brackets,1),1);
    root_brightness = nan(size(brackets,1),1);
    root_eigen_residual = nan(size(brackets,1),1);

    for j = 1:size(brackets,1)
        if brackets(j,1)==brackets(j,2)
            roots_nm(j) = brackets(j,1);
        else
            residual_function = @(lambda_nm) wavelength_residual(lambda_nm, ...
                kernel,r,w,rho,eta_core,n_medium,r_outer_nm,t_shell_nm, ...
                branch_window_nm);
            roots_nm(j) = fzero(residual_function,brackets(j,:));
        end

        [~,root_lambda0(j),root_brightness(j),root_eigen_residual(j)] = ...
            predict_at_material_wavelength(roots_nm(j),kernel,r,w,rho, ...
            eta_core,n_medium,r_outer_nm,t_shell_nm,branch_window_nm);
    end

    % If more than one fixed point exists, report all of them and identify
    % the brightest radial mode without reference to the COMSOL answer.
    [~,selected_root] = max(root_brightness);
    predicted_peak_nm = roots_nm(selected_root);
    lambda0 = root_lambda0(selected_root);

    fprintf('SELF-CONSISTENT CANDIDATES\n');
    for j = 1:numel(roots_nm)
        fprintf(['  #%d: %.4f nm | brightness %.4e | lambda0 ', ...
            '= %.6g %+.6gi | eigen residual %.3e\n'],j,roots_nm(j), ...
            root_brightness(j),real(root_lambda0(j)),imag(root_lambda0(j)), ...
            root_eigen_residual(j));
    end

    fprintf('\nSELECTED BRIGHTEST NIR CANDIDATE\n');
    fprintf('Predicted wavelength = %.4f nm\n',predicted_peak_nm);
    fprintf('lambda0 = %.10g %+.10gi\n',real(lambda0),imag(lambda0));

    if ~isempty(comparison_peak_nm)
        comparison_error_nm = predicted_peak_nm-comparison_peak_nm;
        comparison_error_percent = 100*abs(comparison_error_nm)/comparison_peak_nm;
        fprintf('Post-prediction comparison peak = %.4f nm\n',comparison_peak_nm);
        fprintf('Wavelength error = %+.4f nm (%.3f%%)\n', ...
            comparison_error_nm,comparison_error_percent);
    else
        comparison_error_nm = NaN;
        comparison_error_percent = NaN;
    end

    negative_axis_ratio = abs(imag(lambda0))/max(abs(real(lambda0)),eps);
    if real(lambda0)<0
        fprintf(['WARNING: the selected eigenvalue has negative real part; ', ...
            '|Im/Re| = %.4f.\n'],negative_axis_ratio);
        fprintf(['Treat this as a diagnostic extension of the old code, not ', ...
            'a validated use of the positive high-contrast theorem.\n']);
    end
end

results = struct( ...
    'geometry',struct('r_core_nm',r_core_nm,'t_shell_nm',t_shell_nm, ...
        'r_outer_nm',r_outer_nm,'rho',rho), ...
    'inputs',struct('n_medium',n_medium,'n_silica',n_silica, ...
        'eta_core',eta_core,'branch_window_nm',branch_window_nm), ...
    'scan',table(material_scan_nm,predicted_nm,fixed_point_residual_nm, ...
        brightness,lambda0_scan,eigen_residual), ...
    'roots_nm',roots_nm,'root_lambda0',root_lambda0, ...
    'root_brightness',root_brightness, ...
    'root_eigen_residual',root_eigen_residual);

if ~isempty(roots_nm)
    results.selected = struct('index',selected_root, ...
        'predicted_peak_nm',predicted_peak_nm,'lambda0',lambda0, ...
        'comparison_peak_nm',comparison_peak_nm, ...
        'comparison_error_nm',comparison_error_nm, ...
        'comparison_error_percent',comparison_error_percent);
end

figure('Color','w','Position',[100 100 900 430]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact')
nexttile
plot(material_scan_nm,predicted_nm,'o-','LineWidth',1.4, ...
    'DisplayName','Framework output')
hold on
plot(material_scan_nm,material_scan_nm,'k--','LineWidth',1.2, ...
    'DisplayName','Self-consistency line')
if ~isempty(roots_nm)
    plot(roots_nm,roots_nm,'rp','MarkerSize',11,'MarkerFaceColor','r', ...
        'DisplayName','Fixed point')
end
grid on
xlabel('Wavelength used for Au dispersion (nm)')
ylabel('Framework wavelength (nm)')
title('Answer-agnostic dispersive scan')
legend('Location','best')

nexttile
plot(material_scan_nm,fixed_point_residual_nm,'o-','LineWidth',1.4)
hold on
yline(0,'k--')
if ~isempty(roots_nm)
    xline(roots_nm,'r:')
end
grid on
xlabel('Wavelength used for Au dispersion (nm)')
ylabel('Prediction minus material wavelength (nm)')
title('Fixed-point residual')

function value = wavelength_residual(lambda_nm,kernel,r,w,rho,eta_core, ...
        n_medium,r_outer_nm,t_shell_nm,branch_window_nm)
    predicted_nm = predict_at_material_wavelength(lambda_nm,kernel,r,w, ...
        rho,eta_core,n_medium,r_outer_nm,t_shell_nm,branch_window_nm);
    if ~isfinite(predicted_nm)
        error('No admissible NIR mode at material wavelength %.4f nm.',lambda_nm);
    end
    value = predicted_nm-lambda_nm;
end

function [predicted_nm,lambda0,brightness,residual] = ...
        predict_at_material_wavelength(lambda_material_nm,kernel,r,w,rho, ...
        eta_core,n_medium,r_outer_nm,t_shell_nm,branch_window_nm)
    eps_gold = au_jc_kreibig(lambda_material_nm,t_shell_nm);
    eta_shell = eps_gold/n_medium^2-1;

    eta = eta_shell*ones(size(r));
    eta(r<=rho) = eta_core;
    operator = kernel.*eta.';
    [vectors,values] = eig(operator);
    eigenvalues = diag(values);

    candidate_brightness = [];
    candidate_wavelength = [];
    candidate_lambda0 = [];
    candidate_residual = [];

    for j = 1:numel(eigenvalues)
        mu = eigenvalues(j);
        if ~isfinite(mu) || abs(mu)<1e-12
            continue
        end

        lambda0_j = 1/mu;
        kR = sqrt(lambda0_j);
        if real(kR)<0
            kR = -kR;
        end
        if ~isfinite(kR) || real(kR)<=1e-12
            continue
        end

        wavelength_j = 2*pi*n_medium*r_outer_nm/real(kR);
        if wavelength_j<branch_window_nm(1) || ...
                wavelength_j>branch_window_nm(2)
            continue
        end

        mode = vectors(:,j);
        mode_energy = real(sum(w.*abs(mode).^2.*r.^2));
        if mode_energy<=0 || ~isfinite(mode_energy)
            continue
        end
        brightness_j = abs(sum(w.*mode.*r.^2))^2/mode_energy;
        residual_j = norm(mode-lambda0_j*(operator*mode),inf)/ ...
            max(norm(mode,inf),eps);

        candidate_brightness(end+1,1) = brightness_j; %#ok<AGROW>
        candidate_wavelength(end+1,1) = wavelength_j; %#ok<AGROW>
        candidate_lambda0(end+1,1) = lambda0_j; %#ok<AGROW>
        candidate_residual(end+1,1) = residual_j; %#ok<AGROW>
    end

    if isempty(candidate_brightness)
        predicted_nm = NaN;
        lambda0 = NaN+1i*NaN;
        brightness = NaN;
        residual = NaN;
        return
    end

    [brightness,index] = max(candidate_brightness);
    predicted_nm = candidate_wavelength(index);
    lambda0 = candidate_lambda0(index);
    residual = candidate_residual(index);
end

function eps_gold = au_jc_kreibig(lambda_nm,t_shell_nm)
    % Johnson-Christy optical constants with the same Kreibig correction
    % used by the older validation scripts supplied with this project.
    JC = [0.64 0.92 13.78; 0.77 0.56 11.21; 0.89 0.43 9.519; ...
          1.02 0.35 8.145; 1.14 0.27 7.150; 1.26 0.22 6.350; ...
          1.39 0.17 5.663; 1.51 0.16 5.083; 1.64 0.14 4.542; ...
          1.76 0.13 4.103; 1.88 0.14 3.697; 2.01 0.21 3.272; ...
          2.13 0.29 2.863; 2.26 0.43 2.455; 2.38 0.62 2.081; ...
          2.50 1.04 1.833];

    photon_energy_eV = 1239.84193/lambda_nm;
    if photon_energy_eV<min(JC(:,1)) || photon_energy_eV>max(JC(:,1))
        error('Wavelength %.3f nm lies outside the tabulated Au range.', ...
            lambda_nm);
    end

    n_gold = interp1(JC(:,1),JC(:,2),photon_energy_eV,'pchip');
    k_gold = interp1(JC(:,1),JC(:,3),photon_energy_eV,'pchip');
    eps_bulk = (n_gold+1i*k_gold)^2;

    omega_p = 8.55;
    gamma_bulk = 0.0184;
    v_F = 1.4e6;
    hbar = 6.5821e-16;
    gamma_surface = gamma_bulk+hbar*v_F/(t_shell_nm*1e-9);
    eps_drude_bulk = -omega_p^2/(photon_energy_eV* ...
        (photon_energy_eV+1i*gamma_bulk));
    eps_drude_surface = -omega_p^2/(photon_energy_eV* ...
        (photon_energy_eV+1i*gamma_surface));
    eps_gold = eps_bulk-eps_drude_bulk+eps_drude_surface;
end
