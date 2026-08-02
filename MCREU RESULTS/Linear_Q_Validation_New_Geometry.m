function results = Linear_Q_Validation_New_Geometry()
%LINEAR_Q_VALIDATION_NEW_GEOMETRY Evaluate the two-layer linear ROM at the
%latest COMSOL inverse-design geometry and report the resonance quality factor.

clc;

%% Physical geometry and validation settings
r_core_nm = 71.906;
t_shell_nm = 8.2508;
r_outer_nm = r_core_nm + t_shell_nm;

lambda_reference_nm = 800.0;
n_medium = 1.0;       % Set this to the COMSOL exterior refractive index.
Q_comsol = NaN;       % Replace NaN with the Q extracted from COMSOL.

%% Reduced-order material coefficients
% Keep both layer coefficients explicit. Replace these values with the
% eta_0^k values from the same material-to-contrast mapping used for COMSOL.
eta1 = 1.0;           % Core coefficient, eta_0^1
eta2 = 0.5;           % Gold-shell coefficient, eta_0^2
material_mapping_confirmed = false;
N = 1000;             % Radial quadrature intervals

if ~material_mapping_confirmed
    warning(['eta1 and eta2 are legacy test coefficients. Set them from the ', ...
        'COMSOL permittivity mapping, then set material_mapping_confirmed = true.']);
end

if ~isreal(eta1) || ~isreal(eta2)
    error(['This real-valued integral-equation solve expects real eta1 and ', ...
        'eta2. A lossy-material inverse solve must split real and imaginary parts.']);
end

r1 = r_core_nm / r_outer_nm;
ka_reference = 2*pi*n_medium*r_outer_nm/lambda_reference_nm;

%% Solve the limiting integral equation
[r, w] = trapezoid_rule(N);
u_initial = 0.1*ones(N + 1, 1);
lambda_initial = 2.6;
z_initial = [u_initial; lambda_initial];
operator = two_layer_operator(r, w, r1, eta1, eta2);

options = optimoptions('fsolve', ...
    'Display', 'iter', ...
    'SpecifyObjectiveGradient', true, ...
    'FunctionTolerance', 1e-12, ...
    'StepTolerance', 1e-12, ...
    'MaxFunctionEvaluations', 2e5, ...
    'MaxIterations', 2000);

system_function = @(z) two_layer_system(z, operator, r, w);
[z, ~, exitflag, output] = fsolve(system_function, z_initial, options);

if exitflag <= 0
    warning('fsolve did not report convergence: %s', output.message);
end

u0 = z(1:N + 1);
lambda0 = z(N + 2);

normalization = sqrt(4*pi*sum(w.*(u0.^2).*(r.^2)));
u0 = u0/normalization;
z_normalized = [u0; lambda0];
integral_residual = norm(system_function(z_normalized), inf);

if lambda0 <= 0
    error('The converged lambda0 must be positive to evaluate the resonance branch.');
end

%% First radiative correction and physical scaling
inside = r <= r1;
outside = ~inside;
U1 = 4*pi*sum(w(inside).*u0(inside).*(r(inside).^2));
U2 = 4*pi*sum(w(outside).*u0(outside).*(r(outside).^2));
U0 = U1 + U2;
weighted_overlap = eta1*U1 + eta2*U2;

lambda1 = -1i*(lambda0^(5/2)/(4*pi))*weighted_overlap*U0;

% The reference wavelength fixes the physical scale h through ka = h*k_h.
% Q is then predicted from the imaginary part and is not fitted to COMSOL.
h_leading = ka_reference/real(sqrt(lambda0));
scale_equation = @(h) h*real(sqrt(lambda0 + h*lambda1)) - ka_reference;
try
    h = fzero(scale_equation, [0.05*h_leading, 5*h_leading]);
catch scale_error
    warning('Scale refinement failed (%s). Using the leading-order h.', ...
        scale_error.message);
    h = h_leading;
end

lambda_h = lambda0 + h*lambda1;
k_h = sqrt(lambda_h);
Q_rom = abs(real(k_h))/(2*abs(imag(k_h)));
lambda_rom_nm = 2*pi*n_medium*r_outer_nm/real(h*k_h);
fwhm_rom_nm = lambda_rom_nm/Q_rom;

if isfinite(Q_comsol) && Q_comsol > 0
    Q_error_percent = 100*abs(Q_rom - Q_comsol)/Q_comsol;
else
    Q_error_percent = NaN;
end

%% Report quantities needed for the poster and paper
fprintf('\nLINEAR TWO-LAYER Q VALIDATION\n');
fprintf('Geometry: r_core = %.4f nm, t_shell = %.4f nm, r_outer = %.4f nm\n', ...
    r_core_nm, t_shell_nm, r_outer_nm);
fprintf('Dimensionless interface: r1 = %.9f\n', r1);
fprintf('Material coefficients: eta1 = %.8g, eta2 = %.8g\n', eta1, eta2);
fprintf('lambda0 = %.12g\n', lambda0);
fprintf('lambda1 = %.12g %+.12gi\n', real(lambda1), imag(lambda1));
fprintf('h = %.12g\n', h);
fprintf('lambda_h = %.12g %+.12gi\n', real(lambda_h), imag(lambda_h));
fprintf('k_h = %.12g %+.12gi\n', real(k_h), imag(k_h));
fprintf('Integral-equation residual (inf norm) = %.3e\n', integral_residual);
fprintf('ROM center wavelength = %.4f nm\n', lambda_rom_nm);
fprintf('Q_ROM = %.6g\n', Q_rom);
fprintf('Approximate ROM FWHM = %.4f nm\n', fwhm_rom_nm);
if isfinite(Q_error_percent)
    fprintf('Q_COMSOL = %.6g\n', Q_comsol);
    fprintf('Relative Q error = %.4f %%\n', Q_error_percent);
else
    fprintf('Q_COMSOL not entered; set Q_comsol near the top of this file.\n');
end

%% Mode plot
figure('Color', 'w');
plot(r, u0, 'LineWidth', 2);
xline(r1, '--', 'Core-shell interface', 'LineWidth', 1.2);
xlabel('Normalized radius, r/r_{outer}');
ylabel('Normalized mode, u_0(r)');
title(sprintf('Linear Two-Layer Mode: Q_{ROM} = %.3f', Q_rom));
grid on;

results = struct( ...
    'r_core_nm', r_core_nm, ...
    't_shell_nm', t_shell_nm, ...
    'r_outer_nm', r_outer_nm, ...
    'r1', r1, ...
    'eta1', eta1, ...
    'eta2', eta2, ...
    'lambda0', lambda0, ...
    'lambda1', lambda1, ...
    'h', h, ...
    'lambda_h', lambda_h, ...
    'k_h', k_h, ...
    'Q_rom', Q_rom, ...
    'Q_comsol', Q_comsol, ...
    'Q_error_percent', Q_error_percent, ...
    'lambda_rom_nm', lambda_rom_nm, ...
    'fwhm_rom_nm', fwhm_rom_nm, ...
    'integral_residual', integral_residual, ...
    'r', r, ...
    'u0', u0);
end

function [F, J] = two_layer_system(z, operator, r, w)
N = numel(r) - 1;
u = z(1:N + 1);
lambda = z(N + 2);

F = zeros(N + 2, 1);
F(1:N + 1) = u - lambda*(operator*u);
F(N + 2) = 4*pi*sum(w.*(u.^2).*(r.^2)) - 1;

if nargout > 1
    J = zeros(N + 2, N + 2);
    J(1:N + 1, 1:N + 1) = eye(N + 1) - lambda*operator;
    J(1:N + 1, N + 2) = -(operator*u);
    J(N + 2, 1:N + 1) = (8*pi*w.*u.*(r.^2)).';
end
end

function operator = two_layer_operator(r, w, r1, eta1, eta2)
[ri, rj] = ndgrid(r, r);
denominator = ri.*rj;
kernel = zeros(size(denominator));
nonzero = denominator > 0;
kernel(nonzero) = (2*pi./denominator(nonzero)).*( ...
    ri(nonzero) + rj(nonzero) - abs(ri(nonzero) - rj(nonzero)));

eta = eta2*ones(size(r));
eta(r <= r1) = eta1;
integration_weight = w.*eta.*(r.^2);
operator = kernel.*integration_weight.';
end

function [r, w] = trapezoid_rule(N)
r = linspace(0, 1, N + 1).';
dr = 1/N;
w = dr*ones(N + 1, 1);
w([1, end]) = dr/2;
end
