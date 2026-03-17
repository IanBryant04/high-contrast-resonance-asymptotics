function TripleLayer_Linear
    clc
    clearvars

    %CASE 6:FINDING LAMBDA_H (LINEAR 3 LAYERS)
    % 0 < r < r1      : eta1
    % r1 < r < r2     : eta_mid
    % r2 < r < 1      : eta2

    %parameters
    r1=0.25;
    r2=0.65;        %second boundary radius
    eta1=1.0;
    eta_mid=1.5;    %middle shell susceptibility
    eta2=0.5;

    fprintf('Running Linear Triple-Layer Resonance\n');
    fprintf('r1=%.2f, r2=%.2f, eta1=%.1f, eta_mid=%.1f, eta2=%.1f\n', r1, r2, eta1, eta_mid, eta2);

    % setup grid and integration weights
    N=1000;
    r=linspace(0,1,N)';
    dr=r(2)-r(1);

    w=ones(N,1)*dr;
    w(1)=dr/2; w(end)=dr/2;

    % initial guess
    u0_guess=2.0*ones(N,1);
    lambda0_guess=2.0;
    z0=[u0_guess; lambda0_guess];

    opts=optimoptions('fsolve','Display','iter','TolFun',1e-12,'TolX',1e-12);
    [z, exitflag]=fsolve(@(z) F_system_trilayer(z,r,w,r1,r2,eta1,eta_mid,eta2), z0, opts);

    if exitflag<=0
        warning('Solver did not converge.');
        return;
    end

    u=z(1:N);
    lambda=z(end);

    % normalize u so the volume integral equals 1 (l2 sense)
    radial_integral_sq=sum(w.*(u.^2).*(r.^2));
    norm_factor=sqrt(4*pi*radial_integral_sq);
    u=u/norm_factor;

    % verify the normalization
    check_norm=4*pi*sum(w.*(u.^2).*(r.^2));

    fprintf('\n--- RESULTS ---\n');
    fprintf('Calculated lambda0 = %.6f\n', lambda);
    fprintf('L2 Norm Check      = %.4f (Should be 1.0)\n', check_norm);

    % calculate the correction term lambda_1
    idx_in=(r<=r1);
    idx_mid=(r>r1 & r<=r2);     %NEW: middle shell index
    idx_out=(r>r2);

    U1=4*pi*sum(w(idx_in).*u(idx_in).*(r(idx_in).^2));
    U_mid=4*pi*sum(w(idx_mid).*u(idx_mid).*(r(idx_mid).^2));   %NEW: middle shell integral
    U2=4*pi*sum(w(idx_out).*u(idx_out).*(r(idx_out).^2));

    U0=U1+U_mid+U2;
    term_weighted=(eta1*U1)+(eta_mid*U_mid)+(eta2*U2);  %NEW: eta_mid*U_mid added

    lambda1=-1i*(lambda^(2.5)/(4*pi))*term_weighted*U0;

    fprintf('\n--- CORRECTION TERM VALUES ---\n');
    fprintf('U1      = %.5f\n', U1);
    fprintf('U_mid   = %.5f\n', U_mid);
    fprintf('U2      = %.5f\n', U2);
    fprintf('U0      = %.5f\n', U0);
    fprintf('Calculated lambda1 = %.5fi\n', imag(lambda1));

    fprintf('\n--- FINAL APPROXIMATION ---\n');
    fprintf('lambda_h = %.5f %.5fi * h\n', lambda, imag(lambda1));

    figure
    plot(r, u, 'LineWidth', 2)
    hold on
    xline(r1,'--r');
    xline(r2,'--b');
    grid on
    xlabel('r')
    ylabel('u_0(r)')
    title(['Linear Three-Layer Eigenfunction (r1=', num2str(r1), ', r2=', num2str(r2), ')'])
    legend('u_0(r)',['r_1=',num2str(r1)],['r_2=',num2str(r2)],'Location','best')
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
            % point is in inner core
            idx_0_to_r=(r<=ri);
            idx_r_to_r1=(r>ri & r<=r1);
            idx_r1_to_r2=(r>r1 & r<=r2);   %NEW: middle shell index
            idx_r2_to_1=(r>r2);             %NEW: outer shell index

            term1=eta1*((1/ri)*sum(u_r2_w(idx_0_to_r)) + sum(u_r_w(idx_r_to_r1)));
            term2=eta_mid*sum(u_r_w(idx_r1_to_r2));    %NEW: middle shell contribution
            term3=eta2*sum(u_r_w(idx_r2_to_1));        %NEW: outer shell now term3

            rhs=lambda*(term1+term2+term3);

        elseif ri<=r2  %most important new section -----------------------------------
            %NEW: point is in middle shell (entire branch is new)
            idx_0_to_r1=(r<=r1);
            idx_r1_to_r=(r>r1 & r<=ri);
            idx_r_to_r2=(r>ri & r<=r2);
            idx_r2_to_1=(r>r2);

            term1=eta1*(1/ri)*sum(u_r2_w(idx_0_to_r1));
            term2=eta_mid*((1/ri)*sum(u_r2_w(idx_r1_to_r)) + sum(u_r_w(idx_r_to_r2)));
            term3=eta2*sum(u_r_w(idx_r2_to_1));

            rhs=lambda*(term1+term2+term3);

        else
            % point is in outer shell
            idx_0_to_r1=(r<=r1);
            idx_r1_to_r2=(r>r1 & r<=r2);
            idx_r2_to_r=(r>r2 & r<=ri);
            idx_r_to_1=(r>ri);

            term1=eta1*(1/ri)*sum(u_r2_w(idx_0_to_r1));
            term2=eta_mid*(1/ri)*sum(u_r2_w(idx_r1_to_r2));
            term3=eta2*((1/ri)*sum(u_r2_w(idx_r2_to_r)) + sum(u_r_w(idx_r_to_1)));

            rhs=lambda*(term1+term2+term3);
        end

        F(i)=u(i)-rhs;
    end
    F(N+1)=sum(u_r2_w)-(1/(4*pi));
end