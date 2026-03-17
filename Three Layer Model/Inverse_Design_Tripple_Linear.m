function Inverse_Design_Triple
    clc
    clearvars

    %CASE 7: INVERSE DESIGN LINEAR 3-LAYER FIND eta_mid
    %Configuration
    r1=0.25;
    r2=0.65;        
    eta1=1.0;
    eta2=0.5;

    %Set Bounds here
    lower_bound=0.5;
    upper_bound=3.0;
    num_of_divisions=30;
    eta_eff_vals=linspace(lower_bound, upper_bound, num_of_divisions);
    eta_mid_vals=zeros(size(eta_eff_vals));

    %Setup Grid
    N=500;
    r=linspace(0,1,N)';
    dr=r(2)-r(1);
    w=ones(N,1)*dr; w(1)=dr/2; w(end)=dr/2;

    %z_guess is carried across the sweep.
    %If solver diverges at a specific eta_eff, reset z_guess = [2.0*ones(N,1); 2.5] here
    z_guess=[2.0*ones(N,1); 2.5];

    for j=1:length(eta_eff_vals)

        target_eta_eff=eta_eff_vals(j);
        fprintf('\n--- INVERSE DESIGN ---\n');
        fprintf('Target eta_eff = %.2f\n\n', target_eta_eff);
        fprintf('%-5s %-12s %-12s\n', 'Iter', 'Current eta_mid', 'Error');

        % Initial Guess
        eta_mid_current=1.5;

        tol=1e-6;
        max_iter=50;

        for k=1:max_iter

            opts=optimoptions('fsolve','Display','off','TolFun',1e-10,'TolX',1e-10);
            [z, exitflag]=fsolve(@(z) F_system_trilayer(z,r,w,r1,r2,eta1,eta_mid_current,eta2), z_guess, opts);

            if exitflag<=0
                warning('Solver failed at eta_eff = %.2f', target_eta_eff);
                break;
            end

            u=z(1:N);
            lambda=z(end);
            z_guess=z;

            radial_integral_sq=sum(w.*(u.^2).*(r.^2));
            norm_factor=sqrt(4*pi*radial_integral_sq);
            u=u/norm_factor;

            % Compute Integrals U1, U_mid, U2, U0
            idx_in=(r<=r1);
            idx_mid=(r>r1 & r<=r2);     %NEW middle shell index
            idx_out=(r>r2);
            %NEW: middle shell integral important --------------
            U1=4*pi*sum(w(idx_in).*u(idx_in).*(r(idx_in).^2));
            U_mid=4*pi*sum(w(idx_mid).*u(idx_mid).*(r(idx_mid).^2));   
            U2=4*pi*sum(w(idx_out).*u(idx_out).*(r(idx_out).^2));
            U0=U1+U_mid+U2;

            %eta_mid: eta_eff = (eta1*U1 + eta_mid*U_mid + eta2*U2) / U0
            eta_mid_new=(target_eta_eff*U0 - eta1*U1 - eta2*U2) / U_mid;  %NEW: solving for eta_mid instead of eta1

            % Check Convergence
            err=abs(eta_mid_new-eta_mid_current);
            fprintf('%-5d %-12.6f %-12.6e\n', k, eta_mid_current, err);

            if err<tol
                eta_mid_vals(j)=eta_mid_new;
                break;
            end

            eta_mid_current=eta_mid_new;
        end
    end

    %Plot eta_mid
    figure;
    plot(eta_eff_vals, eta_mid_vals, 'o-', 'LineWidth', 1.5);
    xlabel('\eta_{eff}');
    ylabel('\eta_{mid}');
    title('Case 7: Inverse Design \eta_{mid}(\eta_{eff})');
    grid on;
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
            idx_r1_to_r2=(r>r1 & r<=r2);
            idx_r2_to_1=(r>r2);

            term1=eta1*((1/ri)*sum(u_r2_w(idx_0_to_r)) + sum(u_r_w(idx_r_to_r1)));
            term2=eta_mid*sum(u_r_w(idx_r1_to_r2));
            term3=eta2*sum(u_r_w(idx_r2_to_1));

            rhs=lambda*(term1+term2+term3);

        elseif ri<=r2 %NEW
            % point is in middle shell
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

    % normalization constraint for the solver
    F(N+1)=sum(u_r2_w)-(1/(4*pi));
end