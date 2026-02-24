function DoubleLayer_Nonlinear_Inner
    clc
    clearvars
    
    fprintf('CASE 5:NONLINEAR INNER + LINEAR OUTER\n');
    
    %Nonlinearity in INNER sphere, Linear OUTER shell
    r1=0.5;
    eta1=1.0;  
    beta1=0.5; 
    eta2=1.0;   %Outer layer linear  
    beta2=0.0;  %Outer layer has NO nonlinearity

    fprintf('r1=%.2f, eta1=%.1f, beta1=%.1f, eta2=%.1f, beta2=%.1f\n\n', r1, eta1, beta1, eta2, beta2);

    %grid
    N=1000; 
    r=linspace(0,1,N)';
    dr=r(2)-r(1);
    
    w=ones(N,1)*dr; 
    w(1)=dr/2;
    w(end)=dr/2;

    %Initial guess
    u0_guess=2.0*ones(N,1); 
    lambda0_guess=2.2;
    z0=[u0_guess; lambda0_guess];

    %Solve the nonlinear
    fprintf('Solving nonlinear eigenvalue problem...\n');
    opts=optimoptions('fsolve','Display','iter','TolFun',1e-12,'TolX',1e-12);
    [z,fval,exitflag]=fsolve(@(z) F_system_multilayer(z,r,w,r1,eta1,eta2,beta1,beta2),z0,opts);

    if exitflag<=0
        warning('Solver did not converge. Exit flag = %d', exitflag);
        return;
    end
    fprintf('Solver converged (exitflag=%d, residual=%.2e)\n\n', exitflag, norm(fval));

    %Extract solution
    u=z(1:N);
    lambda=z(end);

    
    radial_integral_sq=sum(w.*(u.^2).*(r.^2));
    norm_factor=sqrt(4*pi*radial_integral_sq);
    u=u/norm_factor;
    
    check_norm=4*pi*sum(w.*(u.^2).*(r.^2));
    fprintf('Normalization: L2 norm = %.6f\n\n', check_norm);
    
    idx_in=(r<=r1);
    idx_out=(r>r1);
    
    %Calculate U_in and U_out
    U_in=4*pi*sum(w(idx_in).*u(idx_in).*(r(idx_in).^2));
    U_out=4*pi*sum(w(idx_out).*u(idx_out).*(r(idx_out).^2));
    U0=U_in+U_out;
    
    %integrated over INNER region only
    U_in_beta=4*pi*sum(w(idx_in).*(u(idx_in).^3).*(r(idx_in).^2));
    
    
    %Compute lambda1
    bracket_term=(eta1*U_in)+(eta2*U_out)+(beta1*U_in_beta);
    lambda1=-1i*(lambda^(2.5)/(4*pi))*bracket_term*U0;
    
    
    %Plot the eigenfunction
    figure('Position',[100,100,800,500]);
    plot(r,u,'LineWidth',2,'Color',[0,0.4470,0.7410])
    hold on
    xline(r1,'--r','LineWidth',1.5); 
    grid on
    xlabel('Radius r','FontSize',12)
    ylabel('u_0(r)','FontSize',12)
    title(sprintf('Nonlinear Inner + Linear Outer Eigenfunction (r_1=%.2f, \\beta_{in}=%.2f)',r1,beta1),'FontSize',13)
    legend('u_0(r)',sprintf('r_1 = %.2f',r1),'Location','best')
    set(gca,'FontSize',11);
end

function F=F_system_multilayer(z,r,w,r1,eta1,eta2,beta1,beta2)    
    N=numel(r);
    u=z(1:N);
    lambda=z(end);
    F=zeros(N+1,1);
    
    u3=u.^3;
    
    %Boundary condition
    F(1)=u(1)-u(2);
    for i=2:N
        ri=r(i);
        
        if ri<=r1
            
            idx_0_to_r=(r<=ri);
            idx_r_to_r1=(r>ri & r<=r1);
            idx_r1_to_1=(r>r1);
            
            
            term_inner=eta1*((1/ri)*sum(w(idx_0_to_r).*u(idx_0_to_r).*r(idx_0_to_r).^2) + ...
                                sum(w(idx_r_to_r1).*u(idx_r_to_r1).*r(idx_r_to_r1))) + ...
                         beta1*((1/ri)*sum(w(idx_0_to_r).*u3(idx_0_to_r).*r(idx_0_to_r).^2) + ...
                                sum(w(idx_r_to_r1).*u3(idx_r_to_r1).*r(idx_r_to_r1)));
            
            term_outer=eta2*sum(w(idx_r1_to_1).*u(idx_r1_to_1).*r(idx_r1_to_1));
            
            rhs=lambda*(term_inner+term_outer);
            
        else
            
            
            idx_0_to_r1=(r<=r1);
            idx_r1_to_r=(r>r1 & r<=ri);
            idx_r_to_1=(r>ri);
            
            term_inner=eta1*(1/ri)*sum(w(idx_0_to_r1).*u(idx_0_to_r1).*r(idx_0_to_r1).^2) + ...
                         beta1*(1/ri)*sum(w(idx_0_to_r1).*u3(idx_0_to_r1).*r(idx_0_to_r1).^2);
            
            term_outer=eta2*((1/ri)*sum(w(idx_r1_to_r).*u(idx_r1_to_r).*r(idx_r1_to_r).^2) + ...
                                sum(w(idx_r_to_1).*u(idx_r_to_1).*r(idx_r_to_1)));
            
            rhs=lambda*(term_inner+term_outer);
        end
        
        F(i)=u(i)-rhs;
    end
    
    F(N+1)=sum(w.*(u.^2).*(r.^2))-(1/(4*pi));
end