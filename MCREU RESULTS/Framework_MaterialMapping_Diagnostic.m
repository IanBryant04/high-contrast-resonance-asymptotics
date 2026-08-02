% Diagnose the material-to-framework mapping without modifying the
% original two-layer eigensolver.
clc
clearvars
close all

%% Geometry and independent material data
r_core_nm = 71.9060;
t_shell_nm = 8.2508;
r_outer_nm = r_core_nm + t_shell_nm;
rho = r_core_nm/r_outer_nm;

lambda_target_nm = 800;
n_medium = 1.0;
n_silica = 1.45;

% Independent optical data at 800 nm. COMSOL is not required to define
% these inputs; use the same literature/material dataset in both models.
eps_core = n_silica^2;
eps_shell = -24.061 - 1.5068i;
eps_medium = n_medium^2;

eta1_physical = eps_core/eps_medium - 1;
eta2_physical = eps_shell/eps_medium - 1;

% The unchanged eigensolver is real-valued. Use its supported lossless
% projection only to diagnose the leading resonance and coefficient map.
eta1_solver = real(eta1_physical);
eta2_solver = real(eta2_physical);
beta1 = 0.0;
beta2 = 0.0;
N = 400;

%% Run the unchanged two-layer solver
r = linspace(0,1,N).';
dr = r(2) - r(1);
w = ones(N,1)*dr;
w([1 end]) = dr/2;
z0 = [2*ones(N,1); 2.2];

opts = optimoptions('fsolve', ...
    'Display','iter', ...
    'FunctionTolerance',1e-12, ...
    'StepTolerance',1e-12, ...
    'MaxFunctionEvaluations',2e5, ...
    'MaxIterations',2000);

timer = tic;
[z,fval,exitflag,output] = fsolve( ...
    @(zz) F_system_multilayer(zz,r,w,rho,eta1_solver,eta2_solver,beta1,beta2), ...
    z0,opts);
runtime_s = toc(timer);
residual = norm(fval,inf);

if exitflag <= 0 || ~isfinite(residual) || residual > 1e-8
    error('Eigensolver failed: exitflag=%d, residual=%.3e.',exitflag,residual);
end

u0 = z(1:N);
lambda0 = z(end);
u0 = u0/sqrt(4*pi*sum(w.*u0.^2.*r.^2));

%% Independent forward-mapping diagnostic
ka_target = 2*pi*n_medium*r_outer_nm/lambda_target_nm;
lambda_target_spectral = ka_target^2;
loss_ratio = abs(imag(eta2_physical))/max(abs(real(eta2_physical)),eps);

fprintf('\nFRAMEWORK MATERIAL-MAPPING DIAGNOSTIC\n');
fprintf('Geometry: core %.4f nm, shell %.4f nm, outer %.4f nm, rho %.9f\n', ...
    r_core_nm,t_shell_nm,r_outer_nm,rho);
fprintf('Independent material inputs at 800 nm:\n');
fprintf('  eta1 = %.8f %+.8fi\n',real(eta1_physical),imag(eta1_physical));
fprintf('  eta2 = %.8f %+.8fi\n',real(eta2_physical),imag(eta2_physical));
fprintf('Real-only solver projection:\n');
fprintf('  eta1_solver = %.8f\n',eta1_solver);
fprintf('  eta2_solver = %.8f\n',eta2_solver);
fprintf('Solver: %.2f s, %d iterations, residual %.3e\n', ...
    runtime_s,output.iterations,residual);
fprintf('lambda0 = %.10f\n',lambda0);
fprintf('800 nm corresponds to positive (kR)^2 = %.10f\n',lambda_target_spectral);
fprintf('Ignored shell-loss ratio |Im(eta2)/Re(eta2)| = %.4f\n',loss_ratio);

if lambda0 > 0
    lambda_leading_nm = 2*pi*n_medium*r_outer_nm/sqrt(lambda0);
    center_error_nm = lambda_leading_nm - lambda_target_nm;
    center_error_percent = 100*abs(center_error_nm)/lambda_target_nm;

    fprintf('\nPASS: the direct map produces a positive spectral branch.\n');
    fprintf('Leading-order ROM wavelength = %.4f nm\n',lambda_leading_nm);
    fprintf('Target difference = %+.4f nm (%.4f%%)\n', ...
        center_error_nm,center_error_percent);
    fprintf(['Q is intentionally not reported: the unchanged real solver omits ', ...
        'the measured material loss.\n']);
    mapping_status = "positive physical branch";
else
    lambda_unsigned_nm = 2*pi*n_medium*r_outer_nm/sqrt(abs(lambda0));

    fprintf('\nFAIL: the direct map does not produce a positive (kR)^2 branch.\n');
    fprintf(['No real forward wavelength or ROM Q can be calculated from this ', ...
        'lambda0.\n']);
    fprintf('Unsigned diagnostic wavelength = %.4f nm (not a prediction).\n', ...
        lambda_unsigned_nm);
    fprintf(['This isolates the issue to the sign/scaling relationship between ', ...
        'eta_0^k and physical permittivity, not fsolve convergence.\n']);
    fprintf(['Check the derivation before deciding whether the framework uses ', ...
        'lambda=(kR)^2, lambda=-(kR)^2, or a high-contrast scaled eta_0^k.\n']);
    mapping_status = "sign/scaling mismatch";
    lambda_leading_nm = NaN;
    center_error_nm = NaN;
    center_error_percent = NaN;
end

results = struct( ...
    'geometry',struct('r_core_nm',r_core_nm,'t_shell_nm',t_shell_nm, ...
        'r_outer_nm',r_outer_nm,'rho',rho), ...
    'materials',struct('eps_core',eps_core,'eps_shell',eps_shell, ...
        'eps_medium',eps_medium,'eta1_physical',eta1_physical, ...
        'eta2_physical',eta2_physical,'eta1_solver',eta1_solver, ...
        'eta2_solver',eta2_solver,'loss_ratio',loss_ratio), ...
    'solver',struct('runtime_s',runtime_s,'iterations',output.iterations, ...
        'exitflag',exitflag,'residual',residual), ...
    'mapping',struct('status',mapping_status,'lambda0',lambda0, ...
        'lambda_target_spectral',lambda_target_spectral, ...
        'lambda_leading_nm',lambda_leading_nm, ...
        'center_error_nm',center_error_nm, ...
        'center_error_percent',center_error_percent), ...
    'r',r,'u0',u0);

figure('Color','w','Position',[120 120 760 430]);
plot(r*r_outer_nm,u0,'LineWidth',2)
hold on
xline(r_core_nm,'--','core-shell interface')
grid on
xlabel('Physical radius (nm)')
ylabel('u_0(r)')
title(sprintf('Real two-layer diagnostic, \\rho = %.3f',rho))

%% Original two-layer eigensolver: copied unchanged for a standalone script
function F = F_system_multilayer(z,r,w,r1,eta1,eta2,beta1,beta2)
    N = numel(r);
    u = z(1:N);
    lambda = z(end);
    F = zeros(N+1,1);
    u3 = u.^3;
    F(1) = u(1)-u(2);

    for i = 2:N
        ri = r(i);
        if ri<=r1
            a = r<=ri;
            b = r>ri & r<=r1;
            c = r>r1;
            term_inner = eta1*((1/ri)*sum(w(a).*u(a).*r(a).^2)+ ...
                sum(w(b).*u(b).*r(b)))+ ...
                beta1*((1/ri)*sum(w(a).*u3(a).*r(a).^2)+ ...
                sum(w(b).*u3(b).*r(b)));
            term_outer = eta2*sum(w(c).*u(c).*r(c));
        else
            a = r<=r1;
            b = r>r1 & r<=ri;
            c = r>ri;
            term_inner = eta1*(1/ri)*sum(w(a).*u(a).*r(a).^2)+ ...
                beta1*(1/ri)*sum(w(a).*u3(a).*r(a).^2);
            term_outer = eta2*((1/ri)*sum(w(b).*u(b).*r(b).^2)+ ...
                sum(w(c).*u(c).*r(c)));
        end
        F(i) = u(i)-lambda*(term_inner+term_outer);
    end
    F(N+1) = sum(w.*u.^2.*r.^2)-1/(4*pi);
end
