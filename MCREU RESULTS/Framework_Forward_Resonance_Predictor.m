function results = Framework_Forward_Resonance_Predictor(config)
% Direct MATLAB port of the Python two-layer resonance evaluator.
% No COMSOL peak or target wavelength enters either prediction.

if nargin < 1
    config = struct();
end
config = merge_config(default_config(),config);
validate_config(config);

R1_nm = config.r_core_nm;
R2_nm = R1_nm + config.t_shell_nm;
rho = R1_nm/R2_nm;
band = config.search_window_nm;
scan_nm = (band(1):config.scan_step_nm:band(2)).';
if scan_nm(end) < band(2), scan_nm(end+1,1) = band(2); end

fprintf('\nFRAMEWORK FORWARD RESONANCE PREDICTOR (PYTHON-FAITHFUL PORT)\n');
fprintf('Geometry: %.4f nm core, %.4f nm shell, %.4f nm outer\n', ...
    R1_nm,config.t_shell_nm,R2_nm);
fprintf('Independent one-shot scan: %.0f-%.0f nm in %.1f nm steps\n', ...
    band(1),band(2),config.scan_step_nm);

leading_curve = nan(size(scan_nm));
corrected_curve = nan(size(scan_nm));
for j = 1:numel(scan_nm)
    fprintf('  Scan %2d/%2d at %.1f nm\n',j,numel(scan_nm),scan_nm(j));
    shot = resonance_oneshot(scan_nm(j),rho,R2_nm,config);
    if shot.converged
        leading_curve(j) = shot.leading_nm;
        corrected_curve(j) = shot.corrected_nm;
    end
end

leading_brackets = sign_change_indices(leading_curve-scan_nm);
corrected_brackets = sign_change_indices(corrected_curve-scan_nm);
fprintf('Found %d leading and %d corrected bracket(s).\n', ...
    numel(leading_brackets),numel(corrected_brackets));

leading_roots = refine_fixed_points(leading_brackets,scan_nm,'leading', ...
    rho,R2_nm,config);
corrected_roots = refine_fixed_points(corrected_brackets,scan_nm,'corrected', ...
    rho,R2_nm,config);
leading = select_longest_root(leading_roots);
corrected = select_longest_root(corrected_roots);

print_result('Leading lambda0 fixed point',leading);
print_result('Corrected lambda_h fixed point',corrected);

comparison = compare_after_prediction(leading,corrected, ...
    config.comparison_peak_nm);
warnings_list = validity_warnings(leading,corrected,config.search_window_nm);
if ~isempty(leading) && ~isempty(corrected)
    status = 'ok_provisional';
else
    status = 'failed';
end

results = struct();
results.status = status;
results.config = config;
results.geometry = struct('r_core_nm',R1_nm, ...
    't_shell_nm',config.t_shell_nm,'r_outer_nm',R2_nm,'rho',rho);
results.leading_roots = leading_roots;
results.corrected_roots = corrected_roots;
results.selected = struct('leading',leading,'corrected',corrected);
results.branch_count = max(numel(leading_roots),numel(corrected_roots));
results.scan = struct('material_nm',scan_nm,'leading_nm',leading_curve, ...
    'corrected_nm',corrected_curve);
results.comparison = comparison;
results.warnings = warnings_list;
results.spectrum = build_resonance_spectrum(leading,band);
if ~isempty(results.spectrum.wavelength_nm)
    fprintf('Plotted ROM peak: %.3f nm | Q %.3f | FWHM %.3f nm\n', ...
        results.spectrum.center_nm,results.spectrum.Q,results.spectrum.fwhm_nm);
end

try
    plot_resonance_spectrum(results.spectrum,config.comparison_peak_nm);
catch exception
    warning('Prediction completed, but plotting failed: %s',exception.message);
end
end

function shot = resonance_oneshot(lambda_nm,rho,R2_nm,config)
% Faithful port of resonance_oneshot(): every call computes a fresh seed.
etas = material_contrasts(lambda_nm,config);
h = 2*pi*config.n_medium*R2_nm/lambda_nm;
[lambda_seed,u_initial,seed_nm,candidate_count] = get_seed( ...
    etas,rho,lambda_nm,R2_nm,config);
if isempty(lambda_seed)
    shot = struct('converged',false,'message','No eigenmode seed found.');
    return
end

[lambda0,u,r,w,converged,residual] = solve_seeded( ...
    rho,etas,lambda_seed,u_initial,config.N_solver);
if ~converged || residual > config.max_solver_residual
    shot = struct('converged',false, ...
        'message',sprintf('Complex fsolve residual %.3e.',residual));
    return
end

[lambda1,W0,D0] = lambda1_correction(lambda0,u,r,w,rho,etas);
lambda_h = lambda0+h*lambda1;
shot = struct('converged',true,'message','', ...
    'material_wavelength_nm',lambda_nm,'seed_nm',seed_nm, ...
    'candidate_count',candidate_count,'lambda0',lambda0, ...
    'lambda1',lambda1,'lambda_h',lambda_h,'h',h,'W0',W0,'D0',D0, ...
    'solver_residual',residual,'r',r,'u',u, ...
    'leading_nm',lambda_to_nm(lambda0,R2_nm,config.n_medium), ...
    'corrected_nm',lambda_to_nm(lambda_h,R2_nm,config.n_medium));
end

function indices = sign_change_indices(residual)
indices = find(isfinite(residual(1:end-1)) & isfinite(residual(2:end)) & ...
    residual(1:end-1).*residual(2:end) <= 0);
end

function roots = refine_fixed_points(indices,scan_nm,kind,rho,R2_nm,config)
roots = [];
for j = 1:numel(indices)
    index = indices(j);
    interval = [scan_nm(index) scan_nm(index+1)];
    fprintf('Refining %s root in %.1f-%.1f nm...\n',kind,interval(1),interval(2));
    objective = @(lambda_nm) oneshot_residual( ...
        lambda_nm,kind,rho,R2_nm,config);
    try
        wavelength_nm = fzero(objective,interval, ...
            optimset('Display','off','TolX',config.fixed_point_tolerance_nm));
        root = resonance_oneshot(wavelength_nm,rho,R2_nm,config);
        root.kind = kind;
        root.wavelength_nm = wavelength_nm;
        if isempty(roots), roots = root; else, roots(end+1) = root; end %#ok<AGROW>
    catch exception
        fprintf('Skipped %s root: %s\n',kind,exception.message);
    end
end
end

function value = oneshot_residual(lambda_nm,kind,rho,R2_nm,config)
shot = resonance_oneshot(lambda_nm,rho,R2_nm,config);
if ~shot.converged, error('%s',shot.message); end
if strcmp(kind,'leading'), value = shot.leading_nm-lambda_nm;
else, value = shot.corrected_nm-lambda_nm; end
end

function root = select_longest_root(roots)
if isempty(roots), root = []; return; end
[~,index] = max([roots.wavelength_nm]);
root = roots(index);
end

function [lambda_seed,u_initial,seed_nm,candidate_count] = get_seed( ...
        etas,rho,lambda_reference_nm,R2_nm,config)
% Faithful port of get_seed(): select the mode nearest the current iterate.
N = config.N_seed;
r_seed = linspace(1e-12,1,N).';
dr = r_seed(2)-r_seed(1);
w_seed = dr*ones(N,1);
w_seed([1 end]) = dr/2;
eta = eta_profile(r_seed,rho,etas);

operator = zeros(N,N);
r2w = r_seed.^2.*w_seed.*eta;
rw = r_seed.*w_seed.*eta;
for j = 1:N
    inside = r_seed <= r_seed(j);
    operator(j,inside) = (r2w(inside)/max(r_seed(j),1e-14)).';
    operator(j,~inside) = rw(~inside).';
end

[vectors,values] = eig(operator);
mu = diag(values);
best_distance = inf;
lambda_seed = [];
best_vector = [];
seed_nm = NaN;
candidate_count = 0;

for j = 1:numel(mu)
    if abs(mu(j)) < 1e-12
        continue
    end
    candidate_lambda = 1/mu(j);
    candidate_nm = lambda_to_nm(candidate_lambda,R2_nm,config.n_medium);
    ratio = abs(real(candidate_lambda))/max(abs(imag(candidate_lambda)),1e-12);
    if candidate_nm < 150 || candidate_nm > 5000 || ratio < 0.2
        continue
    end
    candidate_count = candidate_count+1;
    distance = abs(candidate_nm-lambda_reference_nm);
    if distance < best_distance
        best_distance = distance;
        lambda_seed = candidate_lambda;
        best_vector = vectors(:,j);
        seed_nm = candidate_nm;
    end
end

if isempty(best_vector)
    u_initial = [];
    return
end
[~,largest] = max(abs(best_vector));
best_vector = best_vector/(best_vector(largest)/abs(best_vector(largest)));
[r,w] = make_grid(config.N_solver);
u_initial = interp1(r_seed,real(best_vector),r,'pchip') ...
    + 1i*interp1(r_seed,imag(best_vector),r,'pchip');
normalizer = sqrt(4*pi*sum(w.*u_initial.^2.*r.^2));
u_initial = u_initial/normalizer;
end

function [lambda,u,r,w,converged,residual] = solve_seeded( ...
        rho,etas,lambda_seed,u_initial,N)
[r,w] = make_grid(N);
eta = eta_profile(r,rho,etas);
beta = zeros(N,1);
z0 = [real(u_initial);imag(u_initial);real(lambda_seed);imag(lambda_seed)];
options = optimoptions('fsolve','Display','off','StepTolerance',1e-12, ...
    'FunctionTolerance',1e-12,'MaxIterations',500,'MaxFunctionEvaluations',2e5);
[z,fval,exitflag] = fsolve(@(z) residual_real_imag(z,r,w,eta,beta),z0,options);

u = z(1:N)+1i*z(N+1:2*N);
lambda = z(2*N+1)+1i*z(2*N+2);
normalizer = sqrt(4*pi*sum(w.*u.^2.*r.^2));
u = u/normalizer;
residual = norm(fval,inf);
converged = exitflag > 0;
end

function output = residual_real_imag(z,r,w,eta,beta)
N = numel(r);
u = z(1:N)+1i*z(N+1:2*N);
lambda = z(2*N+1)+1i*z(2*N+2);
F = residual_complex(u,lambda,r,w,eta,beta);
output = [real(F);imag(F)];
end

function F = residual_complex(u,lambda,r,w,eta,beta)
% Vectorized cumsum form copied from F_complex() in the Python prototype.
N = numel(r);
u3 = u.^3;
a = w.*eta.*u.*r.^2;
b = w.*beta.*u3.*r.^2;
c = w.*eta.*u.*r;
d = w.*beta.*u3.*r;
ca = cumsum(a); cb = cumsum(b); cc = cumsum(c); cd = cumsum(d);
inverse_r = zeros(N,1);
inverse_r(2:end) = 1./r(2:end);
inside = inverse_r.*(ca+cb);
outside = (cc(end)-cc)+(cd(end)-cd);
Ffield = u-lambda*(inside+outside);
Ffield(1) = u(1)-u(2);
normalization = sum(w.*u.^2.*r.^2)-1/(4*pi);
F = [Ffield;normalization];
end

function [lambda1,W0,D0] = lambda1_correction(lambda0,u,r,w,rho,etas)
core = (r > 0) & (r <= rho);
shell = r > rho;
U = [4*pi*sum(w(core).*u(core).*r(core).^2); ...
     4*pi*sum(w(shell).*u(shell).*r(shell).^2)];
E = [4*pi*sum(w(core).*u(core).^2.*r(core).^2); ...
     4*pi*sum(w(shell).*u(shell).^2.*r(shell).^2)];
W0 = sum(etas.*U);
D0 = sum(etas.*E);
lambda1 = -1i*(lambda0^2.5/(4*pi))*W0^2/D0;
end

function wavelength_nm = lambda_to_nm(lambda,R2_nm,n_medium)
kR = sqrt(lambda);
if real(kR) <= 0
    kR = -kR;
end
wavelength_nm = 2*pi*n_medium*R2_nm/real(kR);
end

function etas = material_contrasts(lambda_nm,config)
epsilon_gold = au_jc_kreibig(lambda_nm,config.t_shell_nm);
eta_core = (config.n_core/config.n_medium)^2-1;
eta_shell = epsilon_gold/config.n_medium^2-1;
etas = [eta_core;eta_shell];
end

function eta = eta_profile(r,rho,etas)
eta = etas(2)*ones(size(r));
eta(r <= rho) = etas(1);
end

function epsilon_gold = au_jc_kreibig(lambda_nm,shell_nm)
JC = [.64 .92 13.78;.77 .56 11.21;.89 .43 9.519;1.02 .35 8.145; ...
    1.14 .27 7.150;1.26 .22 6.350;1.39 .17 5.663;1.51 .16 5.083; ...
    1.64 .14 4.542;1.76 .13 4.103;1.88 .14 3.697;2.01 .21 3.272; ...
    2.13 .29 2.863;2.26 .43 2.455;2.38 .62 2.081;2.50 1.04 1.833];
energy_eV = 1239.84193/lambda_nm;
n_gold = interp1(JC(:,1),JC(:,2),energy_eV,'pchip');
k_gold = interp1(JC(:,1),JC(:,3),energy_eV,'pchip');
epsilon_bulk = (n_gold+1i*k_gold)^2;
omega_p = 8.55;
gamma_bulk = 0.0184;
gamma_surface = gamma_bulk+6.5821e-16*1.4e6/(shell_nm*1e-9);
drude_bulk = -omega_p^2/(energy_eV*(energy_eV+1i*gamma_bulk));
drude_surface = -omega_p^2/(energy_eV*(energy_eV+1i*gamma_surface));
epsilon_gold = epsilon_bulk-drude_bulk+drude_surface;
end

function [r,w] = make_grid(N)
r = linspace(0,1,N).';
dr = r(2)-r(1);
w = dr*ones(N,1);
w([1 end]) = dr/2;
end

function print_result(label,result)
if isempty(result), fprintf('%s: no fixed point\n',label); return; end
fprintf('%s: %.6f nm | h=%.4f | residual %.2e\n', ...
    label,result.wavelength_nm,result.h,result.solver_residual);
end

function warnings_list = validity_warnings(leading,corrected,band)
warnings_list = {};
for selected = {leading,corrected}
    root = selected{1};
    if isempty(root), warnings_list{end+1} = 'No fixed point was refined.'; continue; end %#ok<AGROW>
    if real(root.lambda0) < 0
        warnings_list{end+1} = 'Negative-real lambda0: wavelength mapping is provisional.'; %#ok<AGROW>
    end
    if root.h >= 0.7
        warnings_list{end+1} = sprintf('h=%.3f is outside the small-h regime.',root.h); %#ok<AGROW>
    end
end
warnings_list{end+1} = sprintf('Independent one-shot solves scanned %.0f-%.0f nm.',band);
warnings_list = unique(warnings_list,'stable');
end

function comparison = compare_after_prediction(leading,corrected,peak_nm)
comparison = struct();
if isempty(peak_nm), return; end
comparison.peak_nm = peak_nm;
if ~isempty(leading)
    comparison.leading_error_nm = leading.wavelength_nm-peak_nm;
    comparison.leading_error_percent = 100*abs(comparison.leading_error_nm)/peak_nm;
end
if ~isempty(corrected)
    comparison.corrected_error_nm = corrected.wavelength_nm-peak_nm;
    comparison.corrected_error_percent = 100*abs(comparison.corrected_error_nm)/peak_nm;
end
end

function spectrum = build_resonance_spectrum(root,band)
% Eigenvalue-plane Q convention used by Full_Validation_Pipeline.m.
spectrum = struct('wavelength_nm',[],'normalized_response',[], ...
    'center_nm',NaN,'fwhm_nm',NaN,'Q',NaN, ...
    'label','Provisional normalized ROM single-pole response');
if isempty(root)
    return
end
Q = abs(real(root.lambda0))/(2*abs(imag(root.lambda0)));
fwhm_nm = root.wavelength_nm/Q;
wavelength_nm = linspace(band(1),band(2),1201).';
response = 1./(1+4*((wavelength_nm-root.wavelength_nm)/fwhm_nm).^2);
spectrum.wavelength_nm = wavelength_nm;
spectrum.normalized_response = response/max(response);
spectrum.center_nm = root.wavelength_nm;
spectrum.fwhm_nm = fwhm_nm;
spectrum.Q = Q;
end

function plot_resonance_spectrum(spectrum,comparison_peak_nm)
if isempty(spectrum.wavelength_nm)
    return
end
figure('Color','w','Position',[120 120 760 480]);
plot(spectrum.wavelength_nm,spectrum.normalized_response, ...
    'LineWidth',2.2,'Color',[0.00 0.35 0.70], ...
    'DisplayName','Reduced-order model');
hold on
xline(spectrum.center_nm,'--','Color',[0.00 0.35 0.70], ...
    'LineWidth',1.4,'DisplayName',sprintf('ROM peak: %.1f nm',spectrum.center_nm));
if ~isempty(comparison_peak_nm)
    xline(comparison_peak_nm,'--','Color',[0.80 0.15 0.12], ...
        'LineWidth',1.4,'DisplayName',sprintf('COMSOL peak: %.1f nm',comparison_peak_nm));
end
grid on
ylim([0 1.05]);
xlabel('Wavelength (nm)');
ylabel('Normalized resonance response');
title(sprintf('ROM resonance: %.1f nm, Q = %.2f, FWHM = %.1f nm', ...
    spectrum.center_nm,spectrum.Q,spectrum.fwhm_nm));
legend('Location','best');
end

function config = default_config()
config = struct('r_core_nm',71.9060,'t_shell_nm',8.2508, ...
    'n_medium',1,'n_core',1.45,'search_window_nm',[500 1000], ...
    'scan_step_nm',10,'N_seed',400,'N_solver',220, ...
    'comparison_peak_nm',[],'fixed_point_tolerance_nm',1e-3, ...
    'max_solver_residual',1e-8);
end

function output = merge_config(defaults,overrides)
output = defaults;
names = fieldnames(overrides);
for j = 1:numel(names)
    if ~isfield(defaults,names{j})
        error('Unknown config field: %s',names{j});
    end
    output.(names{j}) = overrides.(names{j});
end
end

function validate_config(config)
if exist('fsolve','file') ~= 2
    error('Optimization Toolbox fsolve is required.');
end
validateattributes(config.search_window_nm,{'numeric'},{'numel',2,'positive'});
if config.search_window_nm(2) <= config.search_window_nm(1)
    error('search_window_nm must be [minimum maximum].');
end
end
