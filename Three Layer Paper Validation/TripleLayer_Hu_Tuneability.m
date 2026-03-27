function TripleLayer_Hu_Validation
    clc
    clearvars

    fprintf('=======HU 2008 VALIDATION (TRIPLE LAYER)==========\n\n');

    %CASE: tunability (Hu et al. 2008) https://opg.optica.org/oe/fulltext.cfm?uri=oe-16-24-19579
    %gold-silica-gold
    % geometry gotten from (Hu Fig. 2)
    r2 = 0.60;                  % 30nm / 50nm (normalized outer silica radius)
    r1_values = linspace(0.1,0.55,20); % sweep around 15/50 → 25/50

    %material parameters (Hu et al.)
    eps_water  = 1.77;          % background (water)
    eps_silica = 2.04;          % SiO2 shell

    %susceptibility conversion: eta = eps/eps_bg - 1
    eta_mid = eps_silica/eps_water - 1;   % silica → ~0.153

    % gold (Johnson & Christy @ 800nm)
    eps_gold = -24.0 + 1.5i;
    eta_gold = real(eps_gold/eps_water - 1);  % ≈ -14.56

    %layer assignment (gold,silicon,gold)
    eta1 = eta_gold;    % inner gold core
    eta2 = eta_gold;    % outer gold shell

    fprintf('eta_silica = %.4f\n', eta_mid);
    fprintf('eta_gold   = %.4f\n\n', eta_gold);

    N = 500;
    r = linspace(0,1,N)';
    dr = r(2)-r(1);
    
    w = ones(N,1)*dr;
    w(1)=dr/2; w(end)=dr/2;

    % initial guess 
    z_guess = [cos(0.5*pi*r); 1.0];

    lambda_results = zeros(size(r1_values));

    for j = 1:length(r1_values)

        r1 = r1_values(j);

        fprintf('Running case r1 = %.2f\n', r1);

        opts = optimoptions('fsolve','Display','off','TolFun',1e-10,'TolX',1e-10);

        [z, exitflag] = fsolve(@(z) F_system_trilayer(z,r,w,r1,r2,eta1,eta_mid,eta2), z_guess, opts);

        if exitflag <= 0
            warning('Solver failed at r1 = %.2f', r1);
            continue;
        end

        u = z(1:N);
        lambda = z(end);
        z_guess = z;

        % normalize
        radial_integral_sq = sum(w.*(u.^2).*(r.^2));
        u = u / sqrt(4*pi*radial_integral_sq);

        lambda_results(j) = lambda;

        fprintf('lambda0 = %.6f\n\n', lambda);
    end

    %table
    fprintf('\nRESULTS TABLE\n');
    fprintf('%-10s %-10s\n', 'r1', 'lambda0');
    for j = 1:length(r1_values)
        fprintf('%-10.2f %-10.6f\n', r1_values(j), lambda_results(j));
    end

    % plot
    figure
    plot(r1_values, lambda_results, 'o-', 'LineWidth', 2)
    grid on
    xlabel('r_1 (normalized core radius)')
    ylabel('\lambda_0')
    title('Hu 2008 Validation: Tunability Curve')
end

function F=F_system_trilayer(z,r,w,r1,r2,eta1,eta_mid,eta2)

    N=numel(r);
    u=z(1:N);
    lambda=z(end);
    F=zeros(N+1,1);

    u_r2_w=w.*u.*(r.^2);
    u_r_w=w.*u.*r;

    F(1)=u(1)-u(2);

    for i=2:N
        ri=r(i);

        if ri<=r1
            idx_0_to_r=(r<=ri);
            idx_r_to_r1=(r>ri & r<=r1);
            idx_r1_to_r2=(r>r1 & r<=r2);
            idx_r2_to_1=(r>r2);

            term1=eta1*((1/ri)*sum(u_r2_w(idx_0_to_r)) + sum(u_r_w(idx_r_to_r1)));
            term2=eta_mid*sum(u_r_w(idx_r1_to_r2));
            term3=eta2*sum(u_r_w(idx_r2_to_1));

        elseif ri<=r2
            idx_0_to_r1=(r<=r1);
            idx_r1_to_r=(r>r1 & r<=ri);
            idx_r_to_r2=(r>ri & r<=r2);
            idx_r2_to_1=(r>r2);

            term1=eta1*(1/ri)*sum(u_r2_w(idx_0_to_r1));
            term2=eta_mid*((1/ri)*sum(u_r2_w(idx_r1_to_r)) + sum(u_r_w(idx_r_to_r2)));
            term3=eta2*sum(u_r_w(idx_r2_to_1));

        else
            idx_0_to_r1=(r<=r1);
            idx_r1_to_r2=(r>r1 & r<=r2);
            idx_r2_to_r=(r>r2 & r<=ri);
            idx_r_to_1=(r>ri);

            term1=eta1*(1/ri)*sum(u_r2_w(idx_0_to_r1));
            term2=eta_mid*(1/ri)*sum(u_r2_w(idx_r1_to_r2));
            term3=eta2*((1/ri)*sum(u_r2_w(idx_r2_to_r)) + sum(u_r_w(idx_r_to_1)));
        end

        F(i)=u(i)-lambda*(term1+term2+term3);
    end

    F(N+1)=sum(u_r2_w)-(1/(4*pi));
end