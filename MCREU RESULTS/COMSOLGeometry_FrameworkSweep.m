% Standalone multilayer framework sweep for MATLAB Online.
clc
clearvars
close all

% Optional COMSOL data. Leave all four empty for a framework-only run.
comsol_lambda_nm = [];
comsol_sigma_sca_um2 = [];
comsol_sigma_abs_um2 = [];
comsol_sigma_ext_um2 = [];

% Geometry and framework inputs.
r_core_nm = 71.9060;
t_shell_nm = 8.2508;
r_outer_nm = r_core_nm+t_shell_nm;
rho = r_core_nm/r_outer_nm;
lambda_target_nm = 800;
eta1 = 3.171;
eta2 = 1.0;
beta1 = 0.0;
beta2 = 0.0;
N = 1000;

% Original real-valued multilayer eigensolver.
r = linspace(0,1,N).';
dr = r(2)-r(1);
w = ones(N,1)*dr;
w([1 end]) = dr/2;
z0 = [2*ones(N,1);2.2];
opts = optimoptions('fsolve','Display','iter', ...
    'FunctionTolerance',1e-12,'StepTolerance',1e-12);

timer = tic;
[z,fval,exitflag,output] = fsolve( ...
    @(zz) F_system_multilayer(zz,r,w,rho,eta1,eta2,beta1,beta2),z0,opts);
runtime_s = toc(timer);
residual = norm(fval);
if exitflag<=0 || ~isfinite(residual) || residual>1e-8
    error('Eigensolver failed: exitflag=%d, residual=%.3e.',exitflag,residual);
end

u = z(1:N);
lambda0 = z(end);
u = u/sqrt(4*pi*sum(w.*u.^2.*r.^2));

inside = r<=rho;
outside = r>rho;
U_in = 4*pi*sum(w(inside).*u(inside).*r(inside).^2);
U_out = 4*pi*sum(w(outside).*u(outside).*r(outside).^2);
U0 = U_in+U_out;
U_in_beta = 4*pi*sum(w(inside).*u(inside).^3.*r(inside).^2);
E_in = 4*pi*sum(w(inside).*u(inside).^2.*r(inside).^2);
E_out = 4*pi*sum(w(outside).*u(outside).^2.*r(outside).^2);
W0 = eta1*U_in+eta2*U_out+beta1*U_in_beta;
D0 = eta1*E_in+eta2*E_out;

% The h=kr conversion remains provisional until physical scaling is fixed.
h_assumed = 2*pi*r_outer_nm/lambda_target_nm;
lambda1_A = -1i*(lambda0^2.5/(4*pi))*W0*U0;
lambda1_B = -1i*(lambda0^2.5/(4*pi))*W0^2/D0;
lambda_h_A = lambda0+h_assumed*lambda1_A;
lambda_h_B = lambda0+h_assumed*lambda1_B;

lambda_nm = (100:1:1200).';
kR = 2*pi*r_outer_nm./lambda_nm;
lambda_spectral = kR.^2;
A_geo = pi*h_assumed^2;
numerator = h_assumed^3*D0*abs(U0)^2;
Qext_A = (kR/A_geo).*imag(numerator./(lambda_h_A-lambda_spectral));
Qext_B = (kR/A_geo).*imag(numerator./(lambda_h_B-lambda_spectral));

forms = ["A";"B"];
lambda_h = [lambda_h_A;lambda_h_B];
pole_center_nm = 2*pi*r_outer_nm./sqrt(real(lambda_h));
sampled_peak_nm = [sampled_peak(lambda_nm,Qext_A);sampled_peak(lambda_nm,Qext_B)];
fwhm_nm = [estimate_fwhm(lambda_nm,Qext_A);estimate_fwhm(lambda_nm,Qext_B)];
Q = real(lambda_h)./(-2*imag(lambda_h));

results.geometry = struct('r_core_nm',r_core_nm,'t_shell_nm',t_shell_nm, ...
    'r_outer_nm',r_outer_nm,'rho',rho);
results.framework = struct('eta_core',eta1,'eta_shell',eta2, ...
    'beta_core',beta1,'beta_shell',beta2,'h_assumed',h_assumed, ...
    'lambda0',lambda0,'lambda1_A',lambda1_A,'lambda1_B',lambda1_B, ...
    'U_in',U_in,'U_out',U_out,'U0',U0,'E_in',E_in,'E_out',E_out, ...
    'W0',W0,'D0',D0, ...
    'comparison_status',['Provisional: physical nondimensionalization and ' ...
    'dispersive material matching are required before COMSOL comparison.']);
results.solver = table(runtime_s,output.iterations,exitflag,residual);
results.poles = table(forms,lambda_h,pole_center_nm,sampled_peak_nm,fwhm_nm,Q);
results.sweep = table(lambda_nm,lambda_spectral,Qext_A,Qext_B);

fprintf('\nGeometry: %.3f nm core, %.3f nm shell, %.3f nm outer; rho=%.6f\n', ...
    r_core_nm,t_shell_nm,r_outer_nm,rho);
fprintf('Solver: %.2f s, %d iterations, residual %.3e\n', ...
    runtime_s,output.iterations,residual);
fprintf('lambda0 = %.8f\n',lambda0);
disp(results.poles)
fprintf(['NOTE: h=kr is provisional. Physical nondimensionalization and ' ...
    'dispersive material matching are required before comparison with COMSOL.\n']);

figure('Color','w','Position',[80 80 1050 470]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile
plot(lambda_nm,Qext_A,'LineWidth',2,'DisplayName','Form A');
hold on
plot(lambda_nm,Qext_B,'--','LineWidth',2,'DisplayName','Form B');
xline(lambda_target_nm,':',sprintf('%.0f nm target',lambda_target_nm));
grid on
xlabel('Physical wavelength (nm)')
ylabel('Normalized extinction response')
title('Single-pole framework spectrum')
legend('Location','best')

nexttile
plot(r*r_outer_nm,u,'LineWidth',2)
hold on
xline(r_core_nm,'--','core-shell interface')
grid on
xlabel('Physical radius (nm)')
ylabel('u_0(r)')
title(sprintf('Breathing mode, \\rho = %.3f',rho))

function peak_x = sampled_peak(x,y)
    valid = isfinite(x)&isfinite(y);
    x = x(valid);
    y = real(y(valid));
    if isempty(x)
        peak_x = NaN;
    else
        [~,i] = max(y);
        peak_x = x(i);
    end
end

function width = estimate_fwhm(x,y)
    x = x(:);
    y = real(y(:));
    valid = isfinite(x)&isfinite(y);
    x = x(valid);
    y = y(valid);
    [peak,ip] = max(y);
    if numel(x)<3 || peak<=0 || ip==1 || ip==numel(y)
        width = NaN;
        return
    end
    half = peak/2;
    il = find(y(1:ip-1)<=half & y(2:ip)>=half,1,'last');
    jr = find(y(ip:end-1)>=half & y(ip+1:end)<=half,1,'first');
    if isempty(il) || isempty(jr)
        width = NaN;
        return
    end
    ir = ip+jr-1;
    xl = crossing_x(x(il),y(il),x(il+1),y(il+1),half);
    xr = crossing_x(x(ir),y(ir),x(ir+1),y(ir+1),half);
    width = xr-xl;
end

function xc = crossing_x(x1,y1,x2,y2,target)
    if y2==y1
        xc = (x1+x2)/2;
    else
        xc = x1+(target-y1)*(x2-x1)/(y2-y1);
    end
end

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
