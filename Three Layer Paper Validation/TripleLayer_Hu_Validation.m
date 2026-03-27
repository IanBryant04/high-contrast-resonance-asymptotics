function TripleLayer_Hu_Validation
    clc
    clearvars

    % CASE: VALIDATION (Hu et al. 2008) https://opg.optica.org/oe/fulltext.cfm?uri=oe-16-24-19579
    %gold-silica-gold
    % geometry gotten from (Hu Fig. 2)
    r1 = 0.30;        % try 0.30, 0.42, 0.50 see figures
    r2 = 0.60;        %30nm / 50nm
    %material parameters 
    eps_water  = 1.77;
    eps_silica = 2.04;

    %susceptibility conversion
    eta_mid = eps_silica/eps_water - 1;   % silica

    %gold (Johnson & Christy 800nm)
    eps_gold = -24.0 + 1.5i;
    eta_gold = real(eps_gold/eps_water - 1);

    %layer assignment
    eta1 = eta_gold;   % inner gold
    eta2 = eta_gold;   % outer gold

    fprintf('Running Hu 2008 Single Case\n');
    fprintf('r1 = %.2f, r2 = %.2f\n', r1, r2);
    fprintf('eta_gold = %.4f, eta_silica = %.4f\n\n', eta_gold, eta_mid);

    %grid
    N = 1000;
    r = linspace(0,1,N)';
    dr = r(2)-r(1);

    w = ones(N,1)*dr;
    w(1)=dr/2; w(end)=dr/2;

    % initial guess
    u0_guess = cos(0.5*pi*r);
    lambda_guess = 1.0;
    z0 = [u0_guess; lambda_guess];

    opts = optimoptions('fsolve','Display','iter','TolFun',1e-12,'TolX',1e-12);

    [z, exitflag] = fsolve(@(z) F_system_trilayer(z,r,w,r1,r2,eta1,eta_mid,eta2), z0, opts);

    if exitflag <= 0
        warning('Solver did not converge.');
        return;
    end

    u = z(1:N);
    lambda = z(end);

    % normalize
    radial_integral_sq = sum(w.*(u.^2).*(r.^2));
    u = u / sqrt(4*pi*radial_integral_sq);

    check_norm = 4*pi*sum(w.*(u.^2).*(r.^2));

    fprintf('\nRESULTS\n');
    fprintf('lambda0 = %.6f\n', lambda);
    fprintf('L2 Norm = %.4f (should be 1.0)\n', check_norm);

    % compute integrals
    idx_in  = (r<=r1);
    idx_mid = (r>r1 & r<=r2);
    idx_out = (r>r2);

    U1   = 4*pi*sum(w(idx_in).*u(idx_in).*(r(idx_in).^2));
    Umid = 4*pi*sum(w(idx_mid).*u(idx_mid).*(r(idx_mid).^2));
    U2   = 4*pi*sum(w(idx_out).*u(idx_out).*(r(idx_out).^2));

    U0 = U1 + Umid + U2;

    fprintf('\nINTEGRALS\n');
    fprintf('U1   = %.5f\n', U1);
    fprintf('Umid = %.5f\n', Umid);
    fprintf('U2   = %.5f\n', U2);
    fprintf('U0   = %.5f\n', U0);

    % lambda1 (linear case)
    bracket = eta1*U1 + eta_mid*Umid + eta2*U2;
    lambda1 = -1i*(lambda^(2.5)/(4*pi))*bracket*U0;

    fprintf('\nCORRECTION TERM\n');
    fprintf('Imag(lambda1) = %.6f\n', imag(lambda1));

    fprintf('\n--- FINAL ---\n');
    fprintf('lambda_h = %.5f + %.5fi*h\n', lambda, imag(lambda1));

    % plot eigenfunction
    figure
    plot(r,u,'LineWidth',2)
    hold on
    xline(r1,'--r');
    xline(r2,'--b');
    grid on
    xlabel('r')
    ylabel('u_0(r)')
    title(['Hu 2008 Single Case (r1=', num2str(r1), ')'])
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