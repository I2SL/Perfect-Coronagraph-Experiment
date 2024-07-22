%% Crosstalk Matrix
%--------------------------------
% Description: 
% This script computes an estimate of the crosstalk matrix associated with
% the reconfigurable Fourier Zernike mode sorter employed in our
% implementation of the perfect coronagraph. We assume that the crosstalk
% matrix has purely real entries and the estimate is made by fitting a
% least squares optimization problem. 
% 
% The following properties are upheld:
% (a) cross-talk matrix acts on the field (not the intensitites)
% (b) we ensure cross-talk matrix is unitary (Lagrange Multiplier)
% (c) gradient descent used to minimize Frobenius norm loss function
%
% The implementation of this optimization is based on Equations (1) and (2)
% of the paper: 
% 
% "Algorithm of Optimization under Unitary Matrix Constraint
% using Approximate Matrix Exponential", T. Abrudan, J. Eriksson,
% V. Koivunen. (2005).
%
% Related papers can be found by the same authors
%
%--------------------------------
% Author(s): Nico Deshler
% Email(s):  ndeshler@arizona.edu
% Date:      July 22, 2024
%--------------------------------


set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');


rl = 1.22/2;

% Load in the mode intensity data
load('../Data/ModeIntensities.mat')
M = M(:,5:end); % M - 4xN matrix (mode intensities as funciton of source position)
y = y(:,5:end); % y - 1xN vector (position of point source in \vec{r} coordinates)

% Generate zernike expansion coefficients for sampled source positions
n =[0,1,2,2];    %  Radial mode indices
m =[0,-1,0,2];   %  Angular mode indices
x = zeros(size(y));
[t,r] = cart2pol(x',y');
Z = FourierZernike(r,t,n,m).'/sqrt(pi);
Z = real(Z) + imag(Z); % remove parity

% Instantiate cross-talk matrix and gradient descent parameters
mu = 1e-3;          % learning rate
W = eye(4);         % instantiation of cross-talk     


% Run optimization to fit the cross-talk matrix to data
num_steps = 5e4;
for n = 1:num_steps
    L = LossGradient(M,Z,W);
    G = L*W' - W*L';
    W = expm(-mu*G)*W;
end



%% FIGURES
% Plot the theoretical mode intensitities expected under the recovered
% crosstalk matrix estimate and compare to the experimetnally measured mode
% intensities. 
c = [0,0,0;
    1,0,0;
    0,1,0;
    0,0,1];
figure
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')
nexttile(1)
hold on
plot(y/rl,abs(W*Z)'.^2,'LineWidth',1.5)
scatter(y/rl,M,10,'filled')
hold off
xlabel('Source Position $y/\sigma$')
ylabel('Relative Mode Intensity')
axis square
box on
grid on
colororder(c);
title('Cross-Talk Calibration Scan')
legend({'Theory','','','',...
         'Measured','','',''})


nexttile(2)
colormap(slanCM('magma'));
imagesc(abs(W))
cbar=colorbar;
clim([0,1])

title(cbar,'$|\Omega_{ij}|$','interpreter','latex')
axis square
title('Cross-Talk Matrix')
xticks(1:4)
yticks(1:4)
xticklabels({'$\psi_0$','$\psi_1$','$\psi_2$','$\psi_3$'})
yticklabels({'$\psi_0$','$\psi_1$','$\psi_2$','$\psi_3$'})


% Compare the crosstalk matrix figur quality and the relative improvement
% in explaining asymmetry intensity measurements compared to Zero-crosstalk
% model.
figure
tiledlayout(2,2,'TileSpacing','compact','Padding','compact')

nexttile(1)
hold on
plot(y/rl,abs(Z').^2,'LineWidth',1.5)
scatter(y/rl,M,10,'filled')
hold off
xlabel('Source Position $y/\sigma$')
ylabel('Relative Mode Intensity')
axis square
box on
grid on
colororder(c);
title('Mode Intensity Curves')
legend({'Zero Cross-Talk','','','',...
      'Measured','','','',})

nexttile(2)
hold on
plot(y/rl,abs(W*Z)'.^2,'LineWidth',1.5)
scatter(y/rl,M,10,'filled')
hold off
xlabel('Source Position $y/\sigma$')
ylabel('Relative Mode Intensity')
axis square
box on
grid on
colororder(c);
title('Corrected Mode Intensity Curves')
legend({'Cross-Talk Adjusted','','','',...
         'Measured','','','',})

nexttile(3)
plot(y/rl,M' - abs(W*Z)'.^2 ,'LineWidth',1.5)
xlabel('Source Position $y/\sigma$')
ylabel('Cross-Talk Adjusted - Measured')
axis square
box on
grid on
title('Cross-Talk Fit Quality')
legend({'$\psi_0$','$\psi_1$','$\psi_2$','$\psi_3$'},'interpreter','latex')
colororder(c);


nexttile(4)
colormap(gray)
imagesc(abs(W))
cbar=colorbar;
title(cbar,'$|\Omega_{ij}|$','interpreter','latex')
axis square
title('Cross-Talk Matrix')
xticklabels({'$\psi_0$','$\psi_1$','$\psi_2$','$\psi_3$'})
yticklabels({'$\psi_0$','$\psi_1$','$\psi_2$','$\psi_3$'})


function dLdW = LossGradient(M,Z,W)
    % Computes the gradient of the loss for the crosstalk matrix fitting
    % problem.
    % M: [4,N]
    % Z: [4,N]
    % W: [4,4]
    P = W*Z;
    dLdW = -4 * sum( (permute(M,[1,3,2])- permute(P.^2,[1,3,2])) .* permute(P,[1,3,2]) .* permute(Z,[3,1,2]),3);
end



    
