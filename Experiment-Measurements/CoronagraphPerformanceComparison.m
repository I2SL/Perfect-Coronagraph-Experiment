%% Coronagraph Performance Comparison
%--------------------------------
% Description:
% This script compares the theoretical performance of four direct imaging
% coronagraphs in the sub-diffraction star-planet separation regime:
% (1) Theoretical Perfect Coronagraph (fundamental mode nulling)
% (2) Experimental Perfect Coronagraph (4-mode truncation)
% (3) PIAACMC
% (4) Vortex
% 
% For all systems we introduce the background noise model found in the
% experimental setup so as to ensure each are compared on equal footing and
% in real-world circumstances.
%--------------------------------
% Author(s): Nico Deshler
% Email(s):  ndeshler@arizona.edu
% Date:      July 22, 2024
%--------------------------------

addpath('../Utils/')

% Rayleigh limit
rl = 1.22/2;

% LOAD IN NOISE MODEL DETAILS
load('../Data/MeasurementModel.mat')
% noise model parameters
% X,Y;                              % image-plane coordinates
% lambda_0;                         % mean photon rate for signal distribution
% lambda_D;                         % mean photon rate for detector dark clicks
% structured_background;            % structured background
% xtalk_mtx;                        % crosstalk matrix

% LOAD IN CORONAGRAPH OPERATORS (zernike representation)
load('../Data/CoronagraphOperators.mat')
% C_PI;     % PIAACMC coronagraph operator
% C_VC;     % Vortex coronagraph operator
% nmax;     % max dimensionality of hilbert space

% FWHM of possible lambda_0 values
lambda_0_rng = [10^6.25,10^7.65]; % used to compute upper and lower bounds of CRLB curves

% exoplanet location scan
y_e = linspace(min(y),max(y),501);
x_e = zeros(size(y_e));
[th_e,r_e] = cart2pol(x_e,y_e);
b = 1e-3;

% Full Hilbert Space Fourier Zernike Modeset
[n,m] = ZernikeIndices(nmax);

% Fourier Zernike Modes
[Th,R] = cart2pol(X,Y);
FZ = FourierZernike(R(:),Th(:),n,m); % FZ modes over image plane

% Zernike modes used in experimental measurement
n_EX =[0,1,2,2];    %  Radial mode indices
m_EX =[0,-1,0,2];   %  Angular mode indices

% expansion coefficients for field in Zernike basis
ze = conj(FourierZernike(r_e(:),th_e(:),n,m)).'/sqrt(pi);   % planet expansion coeffs
zs = conj(FourierZernike(0,0,n,m)).'/sqrt(pi);              % star expansion coeffs

% cast cross-talk matrix into larger zernike space
temp = zeros(size(C_PI));
for i = 1:size(xtalk_mtx,1)
    for j = 1:size(xtalk_mtx,2)
        entry = xtalk_mtx(i,j);
        row = find((n == n_EX(i)) & (m == m_EX(i)));
        col = find((n == n_EX(j)) & (m == m_EX(j)));
        temp(row,col) = entry;
    end
end
xtalk_mtx = temp;
xtalk_mtx_PC = xtalk_mtx + eye(size(C_PI)).*(xtalk_mtx == 0); 

% Coronagraph Operators (Zernike Representation)
C_PC = diag((n~=0 | m~=0));     % Perfect Coronagraph (Theory)
C_EX = diag((n~=0 | m~=0) & sum((n == n_EX') .* (m == m_EX'),1)); % Perfect Coronagraph (Experimental)

% Get probability distributions at detector for all coronagraphs
d2x = (X(1,2) - X(1,1))^2;  % differential area
P_EX =  ((1-b) * abs( FZ * xtalk_mtx' * C_EX * xtalk_mtx * zs ).^2 ...
          + b  * abs( FZ * xtalk_mtx' * C_EX * xtalk_mtx * ze ).^2 );
P_PC =  ((1-b) * abs( FZ * xtalk_mtx_PC' * C_PC * xtalk_mtx_PC * zs ).^2 ...
          + b  * abs( FZ * xtalk_mtx_PC' * C_PC * xtalk_mtx_PC * ze ).^2 );
P_PI = ((1-b) * abs( FZ * C_PI * zs ).^2 + b* abs( FZ * C_PI * ze ).^2 );
P_VC = ((1-b) * abs( FZ * C_VC * zs ).^2 + b* abs( FZ * C_VC * ze ).^2 );

% measurement distribution of poisson rate
Ns = 1e3;    % star images
Ne = 1;      % exoplanet images
N = Ns + Ne; % total images
background_rate = (structured_background(:) + lambda_D);
lambda_B = d2x*sum(background_rate(:));

% background probability distribution
P_B = background_rate / lambda_B;

% relative weighting of signal to background
w0 = lambda_0 / ( lambda_0 + lambda_B);

% signal-to-noise
exo_photons = lambda_0*b*(1 - abs(FourierZernike(r_e',th_e',0,0)/sqrt(pi)).^2);
SNR = exo_photons./sqrt(exo_photons+lambda_B);

% detection plane probabilities with noise
P_PC_sys = w0 * P_PC + (1-w0) * P_B;
P_EX_sys = w0 * P_EX + (1-w0) * P_B;
P_PI_sys = w0 * P_PI + (1-w0) * P_B;
P_VC_sys = w0 * P_VC + (1-w0) * P_B;

% calculate the CFI (per photon) associated with each system
CFI_PC =  d2x * sum((diff(P_PC_sys,1,2)./diff(y_e,1,2)).^2 ./  P_PC_sys(:,1:end-1),1);
CFI_EX =  d2x * sum((diff(P_EX_sys,1,2)./diff(y_e,1,2)).^2 ./  P_EX_sys(:,1:end-1),1);
CFI_PI =  d2x * sum((diff(P_PI_sys,1,2)./diff(y_e,1,2)).^2 ./  P_PI_sys(:,1:end-1),1);
CFI_VC =  d2x * sum((diff(P_VC_sys,1,2)./diff(y_e,1,2)).^2 ./  P_VC_sys(:,1:end-1),1);

% calculate the total CFI by rescaling the per-photon CFI by the mean photon rate for each off-axis source position
CFI_PC = CFI_PC .* (N*(lambda_0*d2x*sum(P_PC(:,1:end-1),1) + lambda_B));
CFI_EX = CFI_EX .* (N*(lambda_0*d2x*sum(P_EX(:,1:end-1),1) + lambda_B));
CFI_PI = CFI_PI .* (N*(lambda_0*d2x*sum(P_PI(:,1:end-1),1) + lambda_B));
CFI_VC = CFI_VC .* (N*(lambda_0*d2x*sum(P_VC(:,1:end-1),1) + lambda_B));


% calculate the MLE estimator bias using monte-carlo sampling methods
num_mc_samples = 300;
y_e_mle_mc = zeros(num_mc_samples,numel(y_e));
mu = (lambda_0 + lambda_B).*P_EX_sys;
mu_alt = permute(mu,[1,3,2]);
for i = 1:num_mc_samples

    % get a monte-carlo sample from the measurement model 
    % (Poiss approxed as Gaussian)
    mc_sample = normrnd(mu,sqrt(mu));
    
    % calculate the log likelihood (poiss to gaussian approx)
    loglike = -(mc_sample - mu_alt ).^2 ./ mu_alt /2 - .5*log(2*pi*mu_alt);
    loglike = squeeze(sum(loglike,1)).';

    % get the mle estimate index
    [~,mle_id] = max(loglike,[],1);

    % get the mle of the 
    y_e_mle_mc(i,:) = y_e(mle_id(:));
end

y_e_mle_expectation = mean(y_e_mle_mc(1:i,:),1,'omitmissing');
bias = y_e_mle_expectation-y_e;         % MLE bias
bias = smoothdata(bias,'gauss',12);     % apply some smoothing
dbias_dye = diff(bias)./(diff(y_e));    % bias gradient
CRB_EX = (1+dbias_dye) .^2 ./ CFI_EX;   % Biased CRB on estimator variance
CRB_EX = smoothdata(CRB_EX,'gauss',12); % apply some smoothing
expected_err = interp1(y_e,bias,y,'cubic'); % expected error (same as bias)

% calculate the quantum limit
kappa = 1-2*b;
QFI = pi^2*(1-kappa^2)*(1-kappa^2* ( 2*besselj(2,2*pi*y_e)./ (pi * y_e)).^2);

% estimator performance
mean_y_hat = mean(y_hat,2,'omitmissing');
sig_y_hat = std(y_hat,0,2,'omitmissing');
err = mean_y_hat-y;
bias_interp = interp1(y_e,bias,y,'cubic','extrap');
rmse_hat = sqrt(sig_y_hat.^2 + bias_interp.^2); % empirical estimate of rmse

% use Jackknife resampling to get error bars on the estimate of the
% MLE variance
nbs = 100;
for i = 1:size(y_hat,1)
    for j = 1:nbs
        sig_group(j) = std(bootstrp(50,@std,y_hat(i,:)),'omitmissing');
    end
    std_sig_y_hat(i) = mean(sig_group);
end

% get expected uncertainty from CFI
uncertainty = sqrt(CRB_EX);
expected_sig = interp1(y_e(1:end-1),uncertainty,y,'cubic','extrap');

% compute upper and lower bounds of CRLB
sig_bounds = zeros(2,numel(y_e)-1);
for k = 1:2
    w0k = lambda_0_rng(k) / ( lambda_0_rng(k) + lambda_B);
    P_EX_k = w0k * P_EX + (1-w0k) * P_B;
    CFI_EX_k =  d2x * sum((diff(P_EX_k,1,2)./diff(y_e,1,2)).^2 ./  P_EX_k(:,1:end-1),1);
    CFI_EX_k = CFI_EX_k .* (N*(lambda_0_rng(k)*d2x*sum(P_EX(:,1:end-1),1) + lambda_B));
    CRB_EX_k = (1+dbias_dye) .^2 ./ CFI_EX_k;
    CRB_EX_k = smoothdata(CRB_EX_k,'gauss',12);
    sig_k = sqrt(CRB_EX_k);
    sig_bounds(k,:) = sig_k;
end

%% Coronagraph CRLB Performance Comparison
% coronagraph line colors
c = [
    [0.9290 0.6940 0.1250] % Perfect
    [0.7216 0.1216 0.2235] % Experiment
    [0.4660 0.8240 0.1880] % PIAACMC
    [0 0.4470 0.8410]      % Vortex
    ];

figure
t = tiledlayout(1,2,"TileSpacing","compact",'Padding','compact');
nexttile(1)
hold on
%plot(y_e/rl, 1./sqrt( N*lambda_0 * QFI)/rl,'k','LineWidth',1.5)
plot(y_e(1:end-1)/rl, 1./sqrt(CFI_PC)/rl,'Color',c(1,:),'LineWidth',1.5)
plot(y_e(1:end-1)/rl, 1./sqrt(CFI_EX)/rl,'--','Color',c(2,:),'LineWidth',1.5)
plot(y_e(1:end-1)/rl, 1./sqrt(CFI_PI)/rl,'Color',c(3,:),'LineWidth',1.5)
plot(y_e(1:end-1)/rl, 1./sqrt(CFI_VC)/rl,'Color',c(4,:),'LineWidth',1.5)
hold off
xlabel('Exoplanet Position $y_e /\sigma$','interpreter','latex')
ylabel({'$\sqrt{\texttt{CRLB}}/\sigma$ '},'interpreter','latex')
set(gca,'Yscale','log')
xlim([-.8,.8])
%ylim([1e-4,1])
ylim([5e-3,1])
axis square
box on
grid on

nexttile(2)
hold on
%plot(y_e/rl, 1./sqrt( N*lambda_0 * QFI)/rl,'k','LineWidth',1.5)
plot(y_e(1:end-1)/rl, 1./sqrt(CFI_PC)/rl,'Color',c(1,:),'LineWidth',1.5)
plot(y_e(1:end-1)/rl, 1./sqrt(CFI_EX)/rl,'--','Color',c(2,:),'LineWidth',1.5)
plot(y_e(1:end-1)/rl, 1./sqrt(CFI_PI)/rl,'Color',c(3,:),'LineWidth',1.5)
plot(y_e(1:end-1)/rl, 1./sqrt(CFI_VC)/rl,'Color',c(4,:),'LineWidth',1.5)
hold off
leg = legend({'Perfect $(K=\infty)$','Experiment $(K=4)$','PIAACMC','Vortex'},'interpreter','latex');
xlabel('Exoplanet Position $y_e /\sigma$','interpreter','latex')
ylabel({'$\sqrt{\texttt{CRLB}}/\sigma$ '},'interpreter','latex')
set(gca,'Yscale','log')
set(gca,'Xscale','log')
%ylim([1e-4,1])
ylim([5e-3,1])
xlim([0,.8])
axis square
box on
grid on

title(t,{'Coronagraph Performance Comparison (Unbiased Estimator CRLB)',...
         'Under Experimental Background'},'interpreter','latex','FontSize',10.45)

saveas(gcf,'../Figures/SVG/CoronagraphPerformanceComparisonCFI','svg')
saveas(gcf,'../Figures/FIG/CoronagraphPerformanceComparisonCFI','fig')


%% Estimator Performance (Empirical Bias and Imprecision)
figure;
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')

f = [1,2,3,4];
v1 = [-1,-.02;
      +1,-.02;
      +1,+.02;
      -1,+.02];
v2 = [-1,+.02;
      +1,+.02;
      +1,+.04;
      -1,+.04];
v3 = [-1,-.02;
      +1,-.02;
      +1,-.04;
      -1,-.04];

nexttile(1)
hold on
errorbar(y/rl,err/rl,sig_y_hat/rl,'k')
plot(y/rl,err/rl,'k','LineWidth',1.5)
patch('Faces',f,'Vertices',v1,'FaceColor','g','FaceAlpha',.2)
patch('Faces',f,'Vertices',v2,'FaceColor','b','FaceAlpha',.3)
patch('Faces',f,'Vertices',v3,'FaceColor','b','FaceAlpha',.3)
hold off
xlabel('Exoplanet Position $y_{e}/\sigma$','interpreter','latex')
ylabel({'Error $(\check{\bar{y}}_{e}-y_{e})/\sigma$'},'interpreter','latex')
title('Localization Error','interpreter','latex')
xticks(-1:.2:1)
yticks(-.4:.1:.4)
ylim([-.15,.15])
xlim([-.8,.8])
xline(-.6,'--k','LineWidth',1)
xline(+.6,'--k','LineWidth',1)
axis square
grid on
box on
legend({'Empirical Mean','',...
    'abs. error $\leq \sigma/50$',...
    'abs. error $\leq \sigma/25$'...
},'interpreter','latex')


nexttile(2)
hold on
errorbar(y/rl, sig_y_hat/rl, std_sig_y_hat/rl,'k')
%errorbar(y/rl, rmse_hat/rl, std_sig_y_hat/rl,'k')
plot(y_e(1:end-1)/rl, sqrt(CRB_EX)/rl,'r','LineWidth',1.5)
plot(y/rl, sig_y_hat/rl,'LineWidth',1.5,'Color','k')
patch([y_e(1:end-1),fliplr(y_e(1:end-1))]/rl,[sig_bounds(1,:),fliplr(sig_bounds(2,:))]/rl,'r','FaceAlpha',.25)
hold off
xlabel('Exoplanet Position $y_{e}/\sigma$','interpreter','latex')
ylabel({'Imprecision $\check{\sigma}_{e}/\sigma$'},'interpreter','latex')
title('Localization Precision','interpreter','latex')
set(gca,'Yscale','log')
xticks(-1:.2:1)
ylim([1e-3,.5])
xlim([-.8,.8])

xline(-.6,'--k','LineWidth',1)
xline(+.6,'--k','LineWidth',1)
legend({'Empirical Std. Dev.',...
        '$\sqrt{\texttt{CRLB}} \leq \sigma_{e}$','',...
        '$\sqrt{\texttt{CRLB}}$ Range',''},'interpreter','latex')
axis square
grid on
box on

saveas(gcf,'../Figures/SVG/ExperimentalCoronagraph_MLEStatistics','svg')
saveas(gcf,'../Figures/FIG/ExperimentalCoronagraph_MLEStatistics','fig')


%% Estimator Performance (Theoretical Bias and Imprecision
figure 
tiledlayout(1,2)
nexttile(1)
plot(y_e/rl,bias/rl,'k','LineWidth',1.5)
xlabel('Exoplanet Position $y_e/\sigma$','interpreter','latex')
ylabel('Bias $\delta(y_e)/\sigma$','interpreter','latex')
title('MLE Bias','interpreter','latex')
axis square
ylim([-.02,.02])
xlim([-.8,.8])
grid on

nexttile(2)
plot(y_e(1:end-1)/rl,sqrt(CRB_EX)/rl,'k','LineWidth',1.5)
xlabel('Exoplanet Position $y_e/\sigma$','interpreter','latex')
ylabel('$\sqrt{CRLB(y_e)}$','interpreter','latex')
title('Cramer-Rao Lower Bound','interpreter','latex')
set(gca,'yscale','log')
axis square
xlim([-.8,.8])
ylim([5e-3,2e-1])
grid on


