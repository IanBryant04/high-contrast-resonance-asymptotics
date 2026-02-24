function Inverse_Design_adjusted
    clc 
    clearvars

    % CASE 3: INVERSE DESIGN (LINEAR TARGET)

    %Configuration
    eta2 = 1.0;           
    r1 = 0.5;             

    %Set Bounds here   NEW
    lower_bound=0.5;
    upper_bound=3.0;
    num_of_divisions=30;
    eta_eff_vals=linspace(lower_bound, upper_bound, num_of_divisions);
    eta1_vals=zeros(size(eta_eff_vals));

    %Setup Grid 
    N = 500; 
    r = linspace(0,1,N)';
    dr = r(2)-r(1);
    w = ones(N,1)*dr; w(1)=dr/2; w(end)=dr/2;

    for j = 1:length(eta_eff_vals)

        target_eta_eff = eta_eff_vals(j);
        fprintf('\n--- INVERSE DESIGN ---\n');
        fprintf('Target eta_eff = %.2f\n\n', target_eta_eff);
        fprintf('%-5s %-12s %-12s\n', 'Iter', 'Current eta1', 'Error');

        % Initial Guess
        eta1_current = 2.0; 

        tol = 1e-6;
        max_iter = 50;

        z_guess = [2.0*ones(N,1); 2.5];

        for k = 1:max_iter

            opts = optimoptions('fsolve','Display','off','TolFun',1e-10,'TolX',1e-10);
            [z, exitflag] = fsolve(@(z) F_system_multilayer(z, r, w, r1, eta1_current, eta2), z_guess, opts);

            if exitflag <= 0
                warning('Solver failed at eta_eff = %.2f', target_eta_eff);
                break;
            end

            u = z(1:N);
            lambda = z(end);
            z_guess = z; 

            radial_integral_sq = sum(w .* (u.^2) .* (r.^2));
            norm_factor = sqrt(4*pi * radial_integral_sq);
            u = u / norm_factor;

            %Compute Integrals
            idx_in = (r <= r1);
            idx_out = (r > r1);

            U1 = 4 * pi * sum(w(idx_in) .* u(idx_in) .* (r(idx_in).^2));  
            U2 = 4 * pi * sum(w(idx_out) .* u(idx_out) .* (r(idx_out).^2)); 
            U0 = U1 + U2;

            %Equation (Case 3 inverse design)
            eta1_new = (target_eta_eff * U0 - eta2 * U2) / U1;

            err = abs(eta1_new - eta1_current);
            fprintf('%-5d %-12.6f %-12.6e\n', k, eta1_current, err);

            if err < tol
                eta1_vals(j) = eta1_new;
                break;
            end

            eta1_current = eta1_new;
        end
    end

    % Plot eta1(eta_eff)
    figure;
    plot(eta_eff_vals, eta1_vals, 'o-', 'LineWidth', 1.5);
    xlabel('\eta_{eff}');
    ylabel('\eta_1');
    title('Case 3: Inverse Design \eta_1(\eta_{eff})');
    grid on;
end

%#didnt touch
function F = F_system_multilayer(z, r, w, r1, eta1, eta2)
    N = numel(r);
    u = z(1:N);
    lambda = z(end);
    F = zeros(N+1,1);
    
    u_r2_w = w .* u .* (r.^2);
    u_r_w = w .* u .* r;
    
    F(1) = u(1)-u(2);

    for i = 2:N
        ri = r(i);
        if ri <= r1
            idx_0_to_r = (r <= ri);
            idx_r_to_r1 = (r > ri & r <= r1);
            idx_r1_to_1 = (r > r1);
            
            term_inner = eta1*((1/ri)*sum(u_r2_w(idx_0_to_r)) + sum(u_r_w(idx_r_to_r1)));
            term_outer = eta2*sum(u_r_w(idx_r1_to_1));
            rhs = lambda*(term_inner + term_outer);
        else
            idx_0_to_r1 = (r <= r1);
            idx_r1_to_r = (r > r1 & r <= ri);
            idx_r_to_1 = (r > ri);
            
            term_inner = eta1*(1/ri)*sum(u_r2_w(idx_0_to_r1));
            term_outer = eta2*((1/ri)*sum(u_r2_w(idx_r1_to_r)) + sum(u_r_w(idx_r_to_1)));
            rhs = lambda*(term_inner + term_outer);
        end
        F(i) = u(i) - rhs;
    end
    F(N+1) = sum(u_r2_w) - (1/(4*pi));
end
