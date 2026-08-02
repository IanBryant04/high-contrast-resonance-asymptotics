% Recover the effective linear shell coefficient eta2 from a wavelength
% target without changing the original two-layer eigensolver or using COMSOL.
clc
clearvars
close all

%% Inverse-design inputs
r_core_nm = 71.9060;
t_shell_nm = 8.2508;
r_outer_nm = r_core_nm + t_shell_nm;
rho = r_core_nm/r_outer_nm;

lambda_target_nm = 800;
n_medium = 1.0;
n_silica = 1.45;
eta1 = n_silica^2/n_medium^2 - 1;

% eta2 is recovered as a real coefficient of the asymptotic framework.
% It is not assumed to equal the complex physical susceptibility of gold.
eta2_initial = 25;
beta1 = 0.0;
beta2 = 0.0;
N = 400;

% With the outer radius as the reference length, lambda0=(kR)^2.
kR_target = 2*pi*n_medium*r_outer_nm/lambda_target_nm;
lambda0_target = kR_target^2;

%% Radial discretization
r = linspace(0,1,N).';
dr = r(2)-r(1);
w = ones(N,1)*dr;
w([1 end]) = dr/2;

options = optimoptions('fsolve', ...
    'Display','iter', ...
    'FunctionTolerance',1e-11, ...
    'StepTolerance',1e-11, ...
    'MaxFunctionEvaluations',1e5, ...
    'MaxIterations',1000);

% Unknowns are the N samples of u0 and the single coefficient eta2.
% lambda0 is fixed by the requested 800 nm resonance.
z0_inverse = [2*ones(N,1); eta2_initial];

fprintf('\nLINEAR ETA2 INVERSE DESIGN\n');
fprintf('Geometry: core %.4f nm, shell %.4f nm, outer %.4f nm\n', ...
    r_core_nm,t_shell_nm,r_outer_nm);
fprintf('Target wavelength = %.4f nm\n',lambda_target_nm);
fprintf('Fixed lambda0=(kR)^2 = %.10f\n',lambda0_target);
fprintf('Known eta1 = %.8f\n',eta1);
fprintf('Initial eta2 guess = %.8f\n\n',eta2_initial);

timer = tic;
[z_inverse,f_inverse,exitflag_inverse,output_inverse] = fsolve( ...
    @(zz) F_inverse_eta2(zz,lambda0_target,r,w,rho,eta1,beta1,beta2), ...
    z0_inverse,options);
runtime_inverse_s = toc(timer);

inverse_residual = norm(f_inverse,inf);
if exitflag_inverse <= 0 || ~isfinite(inverse_residual) || ...
        inverse_residual > 1e-8
    error('Inverse solve failed: exitflag=%d, residual=%.3e.', ...
        exitflag_inverse,inverse_residual);
end

u0_inverse = z_inverse(1:N);
eta2_effective = z_inverse(end);
u0_inverse = u0_inverse/sqrt(4*pi*sum(w.*u0_inverse.^2.*r.^2));

%% Unchanged forward-solver check
% The original eigensolver is now given the recovered eta2. Starting from
% the inverse mode checks that it independently returns lambda0_target.
z0_forward = [u0_inverse;lambda0_target];
timer = tic;
[z_forward,f_forward,exitflag_forward,output_forward] = fsolve( ...
    @(zz) F_system_multilayer(zz,r,w,rho,eta1,eta2_effective,beta1,beta2), ...
    z0_forward,options);
runtime_forward_s = toc(timer);

forward_residual = norm(f_forward,inf);
if exitflag_forward <= 0 || ~isfinite(forward_residual) || ...
        forward_residual > 1e-8
    error('Forward check failed: exitflag=%d, residual=%.3e.', ...
        exitflag_forward,forward_residual);
end

u0_forward = z_forward(1:N);
lambda0_forward = z_forward(end);
u0_forward = u0_forward/sqrt(4*pi*sum(w.*u0_forward.^2.*r.^2));

if lambda0_forward <= 0
    error('The forward check returned a nonpositive resonance branch.');
end

lambda_rom_nm = 2*pi*n_medium*r_outer_nm/sqrt(lambda0_forward);
center_error_nm = lambda_rom_nm-lambda_target_nm;
center_error_percent = 100*abs(center_error_nm)/lambda_target_nm;

fprintf('\nINVERSE RESULT\n');
fprintf('eta2_effective = %.10f\n',eta2_effective);
fprintf('Inverse residual = %.3e\n',inverse_residual);
fprintf('Inverse runtime = %.2f s (%d iterations)\n', ...
    runtime_inverse_s,output_inverse.iterations);

fprintf('\nUNCHANGED FORWARD CHECK\n');
fprintf('lambda0 = %.10f\n',lambda0_forward);
fprintf('ROM wavelength = %.6f nm\n',lambda_rom_nm);
fprintf('Center error = %+.6f nm (%.6g%%)\n', ...
    center_error_nm,center_error_percent);
fprintf('Forward residual = %.3e\n',forward_residual);
fprintf('Forward runtime = %.2f s (%d iterations)\n', ...
    runtime_forward_s,output_forward.iterations);

if eta2_effective <= 0
    fprintf(['WARNING: recovered eta2 is not on the positive branch assumed ', ...
        'by the current resonance interpretation.\n']);
end

fprintf(['\neta2_effective is an inverse-designed framework coefficient, ', ...
    'not a direct gold permittivity value.\n']);
fprintf(['Q is intentionally omitted: the unchanged real solver contains ', ...
    'neither material loss nor a validated h scaling.\n']);

results = struct( ...
    'geometry',struct('r_core_nm',r_core_nm,'t_shell_nm',t_shell_nm, ...
        'r_outer_nm',r_outer_nm,'rho',rho), ...
    'target',struct('lambda_nm',lambda_target_nm,'kR',kR_target, ...
        'lambda0',lambda0_target), ...
    'inverse',struct('eta1',eta1,'eta2_effective',eta2_effective, ...
        'residual',inverse_residual,'exitflag',exitflag_inverse, ...
        'iterations',output_inverse.iterations,'runtime_s',runtime_inverse_s), ...
    'forward_check',struct('lambda0',lambda0_forward, ...
        'lambda_rom_nm',lambda_rom_nm,'center_error_nm',center_error_nm, ...
        'center_error_percent',center_error_percent, ...
        'residual',forward_residual,'exitflag',exitflag_forward, ...
        'iterations',output_forward.iterations,'runtime_s',runtime_forward_s), ...
    'r',r,'u0',u0_forward);

figure('Color','w','Position',[120 120 760 430]);
plot(r*r_outer_nm,u0_forward,'LineWidth',2)
hold on
xline(r_core_nm,'--','core-shell interface')
grid on
xlabel('Physical radius (nm)')
ylabel('u_0(r)')
title(sprintf('Inverse-designed mode, eta_2 = %.4f',eta2_effective))

function F = F_inverse_eta2(z,lambda0_target,r,w,r1,eta1,beta1,beta2)
    N = numel(r);
    u = z(1:N);
    eta2 = z(end);

    % Reuse the unchanged forward residual with lambda fixed and eta2 free.
    F = F_system_multilayer([u;lambda0_target],r,w,r1, ...
        eta1,eta2,beta1,beta2);
end

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
