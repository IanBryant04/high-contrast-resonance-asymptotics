function TripleLayer_Hu_Qext
    clc
    clearvars

    %=== Hu et al. 2008 Fig.2: Q_ext vs wavelength ===
    %    Quasistatic polarizability (Bohren&Huffman 1983)
    %    + MLWA correction (Meier&Wokaun 1983)
    %    Gold: J&C Table I (Phys.Rev.B 6, 1972) — direct from paper

    % J&C Table I gold data — transcribed from uploaded paper
    % eV->nm via lam=1239.8/eV, only 400-1000nm range kept
    JC_lam=[397.4;413.3;430.5;450.8;471.4;495.9;520.9;548.6;582.1;616.8;659.5;704.4;756.0;821.1;891.9;984.0];
    JC_n=[1.470;1.460;1.450;1.380;1.310;1.040;0.620;0.430;0.290;0.210;0.140;0.130;0.140;0.160;0.170;0.220];
    JC_k=[1.952;1.958;1.948;1.914;1.849;1.833;2.081;2.455;2.863;3.272;3.697;4.103;4.542;5.083;5.663;6.350];
    JC_eps_real=JC_n.^2-JC_k.^2;
    JC_eps_imag=2*JC_n.*JC_k;

    % material constants (Hu Sec.2)
    eps_water=1.77;
    eps_silica=2.04;

    % wavelength sweep + pchip interpolation of gold eps
    lam=linspace(400,1000,200)';
    eg_re=interp1(JC_lam,JC_eps_real,lam,'pchip');
    eg_im=interp1(JC_lam,JC_eps_imag,lam,'pchip');

    % Hu Fig.2 geometries: R1/R2/R3 (nm)
    R1_all=[25,21,15,0];
    R2_all=[30,30,30,30];
    R3_all=[50,50,50,50];
    labels={'R25/30/50 nm','R21/30/50 nm','R15/30/50 nm','R30/50 nm'};
    colors={'b',[0 0.6 0],'r','k'};

    % Hu peaks read from Fig.2 (~+/-10nm uncertainty)
    hu_peak_lam=[800,680,640,620];

    Qext_all=zeros(length(lam),4);

    for c=1:4
        R1=R1_all(c); R2=R2_all(c); R3=R3_all(c);

        for j=1:length(lam)
            eg=eg_re(j)+1i*eg_im(j);

            % recursive effective permittivity (Bohren&Huffman Ch.5)
            if R1>0
                % 3-layer: gold/silica/gold
                f1=(R1/R2)^3;
                eps_12=eps_silica*(eg*(1+2*f1)+2*eps_silica*(1-f1))/...
                                  (eg*(1-f1)+eps_silica*(2+f1));
                f2=(R2/R3)^3;
                eps_eff=eg*(eps_12*(1+2*f2)+2*eg*(1-f2))/...
                           (eps_12*(1-f2)+eg*(2+f2));
            else
                % 2-layer CNS: silica/gold
                f=(R2/R3)^3;
                eps_eff=eg*(eps_silica*(1+2*f)+2*eg*(1-f))/...
                           (eps_silica*(1-f)+eg*(2+f));
            end

            K0=(eps_eff-eps_water)/(eps_eff+2*eps_water);
            x=2*pi*sqrt(eps_water)*R3/lam(j);

            %NEW: MLWA correction — analogous to Im(lambda_1) in Meklachi
            K_dyn=K0/(1-x^2*K0-1i*(2/3)*x^3*K0);

            Qext_all(j,c)=4*x*imag(K_dyn);
        end
    end

    Qext_all=max(Qext_all,0);

    % extract peaks
    our_peak_lam=zeros(1,4);
    our_peak_Qext=zeros(1,4);
    for c=1:4
        [pk,idx]=max(Qext_all(:,c));
        our_peak_lam(c)=lam(idx);
        our_peak_Qext(c)=pk;
    end

    %--- plot 1: Q_ext spectra ---
    figure('Position',[100 100 700 500])
    hold on
    for c=1:4
        plot(lam,Qext_all(:,c),'Color',colors{c},'LineWidth',2)
    end
    xlabel('Wavelength (nm)')
    ylabel('Extinction efficiency, Q_{ext}')
    legend(labels{:},'Location','northeast')
    xlim([400 1000])
    grid on
    title('Q_{ext} via Quasistatic+MLWA — Hu et al. (2008) Fig. 2 Geometry')
    set(gca,'FontSize',11)
    hold off

    %--- plot 2: error bar chart ---
    figure('Position',[100 600 600 400])
    err_pct=100*(our_peak_lam-hu_peak_lam)./hu_peak_lam;
    bar_colors=[0 0 1;0 0.6 0;1 0 0;0 0 0];
    b=bar(err_pct);
    b.FaceColor='flat';
    for c=1:4
        b.CData(c,:)=bar_colors(c,:);
    end
    set(gca,'XTickLabel',labels,'FontSize',10)
    ylabel('Peak wavelength error (%)')
    title('Error vs Hu et al. (2008) Full Mie Theory')
    grid on
    hold on
    yline(0,'k--','LineWidth',1);
    yline(5,'r--','5% threshold');
    yline(-5,'r--','-5% threshold');
    hold off

    %--- plot 3: peak wavelength comparison ---
    figure('Position',[700 600 500 400])
    plot(1:4,hu_peak_lam,'ks-','MarkerSize',10,'LineWidth',2,'MarkerFaceColor','k')
    hold on
    for c=1:4
        plot(c,our_peak_lam(c),'o','MarkerSize',10,'LineWidth',2,...
            'Color',bar_colors(c,:),'MarkerFaceColor',bar_colors(c,:))
    end
    set(gca,'XTick',1:4,'XTickLabel',labels,'FontSize',9)
    ylabel('Peak wavelength (nm)')
    title('Peak Position: Our MLWA (circles) vs Hu Mie (squares)')
    legend('Hu (Mie)','Ours (MLWA)','Location','northwest')
    grid on
    hold off
end
