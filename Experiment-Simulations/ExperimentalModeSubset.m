%% Experimental Fourier-Zernike Mode Subset
%--------------------------------
% Description: 
% This script computes the probability of photon detection in each mode of
% the truncated Fourier-Zernike basis (modes 0-4) used in our experimental
% setup as a function of star-planet separation. It also calculates the 
% classical fisher information supplied by each mode for exoplanet 
% localization compared to the quantum fisher information limit. These
% curves are plotted at the end of the script
%--------------------------------
% Author(s): Nico Deshler
% Email(s):  ndeshler@arizona.edu
% Date:      July 22, 2024
%--------------------------------

addpath('../Utils/')

% Experimental Subset of Fourier-Zernike modes
rl = 1.22/2;
n = [0,1,2,2];
m = [0,-1,0,2];
b = 1e-3;
delta_p = 1-2*b;

% off-axis source position
y = rl*linspace(0,1,1000)';
x = zeros(size(y));
[th,r] = cart2pol(x,y);

% get mode probabilities
p_nm = abs(FourierZernike(r,th,n,m)).^2/pi;

% Get radial CFI
[CFI_nm_rr,~,~,~,~,~] = StarPlanet_FTZernikeCFIM(r,th,n,m,b);

% QFI radial component
QFI_rr = (1-delta_p.^2)*pi^2 - 4*(1-delta_p.^2).*delta_p.^2 ...
                            .*(besselj(2,2*pi*r)./(r)).^2; 


%% FIGURES
% subfig 1: Mode Probabilities
figure;
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')
nexttile(1)
plot(y/rl,p_nm,'--','LineWidth',1.5)
hold on
plot(y/rl,sum(p_nm,2),'k','LineWidth',1.5)
plot(y/rl,ones(size(y)),'-.k','LineWidth',1.5)
hold off
xlabel('Exoplanet Position $y_{e}/\sigma$','interpreter','latex')
ylabel('Photon Mode Probability $P(y_e)$','interpreter','latex')
leg_names = arrayfun(@(j) sprintf('$ P_{ %d }$',j),0:(numel(n)-1),'UniformOutput',false);
leg = legend([leg_names,'$P = \sum_{k=0}^{3} P_{k}$','P=1'],'interpreter','latex');
title(leg,'Mode','interpreter','latex')
title('Truncated Basis Support','interpreter','latex')
axis square
box on
ylim([0,1.2])

% subfig 1: Mode CFIs
nexttile(2)
plot(y/rl,CFI_nm_rr/((1-delta_p.^2)*pi^2),'--','LineWidth',1.5)
hold on
plot(y/rl,sum(CFI_nm_rr,2)/((1-delta_p.^2)*pi^2),'k','LineWidth',1.5)
plot(y/rl,QFI_rr/((1-delta_p.^2)*pi^2),'-.k','LineWidth',1.5)
hold off

xlabel('Exoplanet Position $y_{e}/\sigma$','interpreter','latex')
ylabel('Fisher Information $\mathcal{I}(y_e) / 4 \pi^2 b(1-b) $','interpreter','latex')
leg_names = arrayfun(@(j) sprintf('$ \\mathcal{I}_{%d}$',j),0:(numel(n)-1),'UniformOutput',false);
leg = legend([leg_names,'CFI $\mathcal{I} = \sum_{k=0}^{3}\mathcal{I}_{k}$','QFI $\mathcal{K}$'],'interpreter','latex');
title(leg,'Mode','interpreter','latex')
title('Truncated Basis Fisher Information','interpreter','latex')
axis square
box on
ylim([0,1.2])

saveas(gcf,'../Figures/SVG/FourModeSubset_Prob_CFI','svg')
saveas(gcf,'../Figures/FIG/FourModeSubset_Prob_CFI','fig')