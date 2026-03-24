function TripleLayer_Nonlinear
    clc
    clearvars

    % CASE 8: NONLINEAR 3 LAYERS
    % 0 < r < r1 : eta1, beta1
    % r1 < r < r2 : eta_mid, beta_mid
    % r2 < r < 1 : eta2, beta2
    % parameters
    r1 = 0.3;
    r2 = 0.6;

    eta1 = 1.0;     beta1 = 0.0;
    eta_mid = 1.5;  beta_mid = 0.3;
    eta2 = 0.5;     beta2 = 0.3;

    fprintf('Running Nonlinear Triple-Layer Resonance\n');

    % grid
    N = 1000;
    r = linspace(0,1,N)';
    dr = r(2)-r(1);

    % weights
    w = ones(N,1)*dr;
    w(1)=dr/2; w(end)=dr/2;

    % initial guess
    u0_guess = 2.0*ones(N,1);
    lambda0_guess = 2.0;
    z0 = [u0_guess; lambda0_guess];

    opts = optimoptions('fsolve','Display','iter','TolFun',1e-12,'TolX',1e-12);

    [z, exitflag] = fsolve(@(z) F_system_trilayer_nl(z,r,w,r1,r2,eta1,eta_mid,eta2,beta1,beta_mid,beta2), z0, opts);

    if exitflag <= 0
        warning('Solver did not converge.');
        return;
    end

    u = z(1:N);
    lambda = z(end);

    % normalize
    radial_integral_sq = sum(w.*(u.^2).*(r.^2));
    norm_factor = sqrt(4*pi*radial_integral_sq);
    u = u / norm_factor;

    check_norm = 4*pi*sum(w.*(u.^2).*(r.^2));

    fprintf('\n--- RESULTS ---\n');
    fprintf('lambda0 = %.6f\n', lambda);
    fprintf('L2 Norm = %.4f\n', check_norm);

    idx_in  = (r<=r1);
    idx_mid = (r>r1 & r<=r2);
    idx_out = (r>r2);

    % linear integrals
    U1   = 4*pi*sum(w(idx_in).*u(idx_in).*(r(idx_in).^2));
    U_mid= 4*pi*sum(w(idx_mid).*u(idx_mid).*(r(idx_mid).^2));
    U2   = 4*pi*sum(w(idx_out).*u(idx_out).*(r(idx_out).^2));

    % nonlinear integrals
    u3 = u.^3;
    U1b   = 4*pi*sum(w(idx_in).*u3(idx_in).*(r(idx_in).^2));
    U_midb= 4*pi*sum(w(idx_mid).*u3(idx_mid).*(r(idx_mid).^2));
    U2b   = 4*pi*sum(w(idx_out).*u3(idx_out).*(r(idx_out).^2));

    U0 = U1 + U_mid + U2;

    bracket = (eta1*U1)+(eta_mid*U_mid)+(eta2*U2) + ...
              (beta1*U1b)+(beta_mid*U_midb)+(beta2*U2b);

    lambda1 = -1i*(lambda^(2.5)/(4*pi))*bracket*U0;

    fprintf('\n--- lambda1 ---\n');
    fprintf('Imag(lambda1) = %.6f\n', imag(lambda1));

    fprintf('\n--- FINAL ---\n');
    fprintf('lambda_h = %.5f + %.5fi*h\n', lambda, imag(lambda1));

    % plot
    figure
    plot(r,u,'LineWidth',2)
    hold on
    xline(r1,'--r'); xline(r2,'--b');
    grid on
    xlabel('r'); ylabel('u(r)')
    title('Nonlinear Triple-Layer Eigenfunction')
end

function F = F_system_trilayer_nl(z,r,w,r1,r2,eta1,eta_mid,eta2,beta1,beta_mid,beta2)

    N = numel(r);
    u = z(1:N);
    lambda = z(end);

    F = zeros(N+1,1);

    % precompute
    u3 = u.^3;

    u_r2_w  = w.*u.*(r.^2);
    u_r_w   = w.*u.*r;

    u3_r2_w = w.*u3.*(r.^2);
    u3_r_w  = w.*u3.*r;

    F(1) = u(1)-u(2);

    for i = 2:N
        ri = r(i);

        if ri <= r1
            idx_0_to_r = (r<=ri);
            idx_r_to_r1 = (r>ri & r<=r1);
            idx_r1_to_r2 = (r>r1 & r<=r2);
            idx_r2_to_1 = (r>r2);

            % linear
            L1 = eta1*((1/ri)*sum(u_r2_w(idx_0_to_r)) + sum(u_r_w(idx_r_to_r1)));
            L2 = eta_mid*sum(u_r_w(idx_r1_to_r2));
            L3 = eta2*sum(u_r_w(idx_r2_to_1));

            % nonlinear
            N1 = beta1*((1/ri)*sum(u3_r2_w(idx_0_to_r)) + sum(u3_r_w(idx_r_to_r1)));
            N2 = beta_mid*sum(u3_r_w(idx_r1_to_r2));
            N3 = beta2*sum(u3_r_w(idx_r2_to_1));

        elseif ri <= r2
            idx_0_to_r1 = (r<=r1);
            idx_r1_to_r = (r>r1 & r<=ri);
            idx_r_to_r2 = (r>ri & r<=r2);
            idx_r2_to_1 = (r>r2);

            L1 = eta1*(1/ri)*sum(u_r2_w(idx_0_to_r1));
            L2 = eta_mid*((1/ri)*sum(u_r2_w(idx_r1_to_r)) + sum(u_r_w(idx_r_to_r2)));
            L3 = eta2*sum(u_r_w(idx_r2_to_1));

            N1 = beta1*(1/ri)*sum(u3_r2_w(idx_0_to_r1));
            N2 = beta_mid*((1/ri)*sum(u3_r2_w(idx_r1_to_r)) + sum(u3_r_w(idx_r_to_r2)));
            N3 = beta2*sum(u3_r_w(idx_r2_to_1));

        else
            idx_0_to_r1 = (r<=r1);
            idx_r1_to_r2 = (r>r1 & r<=r2);
            idx_r2_to_r = (r>r2 & r<=ri);
            idx_r_to_1 = (r>ri);

            L1 = eta1*(1/ri)*sum(u_r2_w(idx_0_to_r1));
            L2 = eta_mid*(1/ri)*sum(u_r2_w(idx_r1_to_r2));
            L3 = eta2*((1/ri)*sum(u_r2_w(idx_r2_to_r)) + sum(u_r_w(idx_r_to_1)));

            N1 = beta1*(1/ri)*sum(u3_r2_w(idx_0_to_r1));
            N2 = beta_mid*(1/ri)*sum(u3_r2_w(idx_r1_to_r2));
            N3 = beta2*((1/ri)*sum(u3_r2_w(idx_r2_to_r)) + sum(u3_r_w(idx_r_to_1)));
        end

        rhs = lambda*(L1+L2+L3 + N1+N2+N3);
        F(i) = u(i) - rhs;
    end

    F(N+1) = sum(w.*(u.^2).*(r.^2)) - (1/(4*pi));
end