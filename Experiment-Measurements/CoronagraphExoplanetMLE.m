%% Coronagraph Exoplanet Maximum Likelihood Estimation
%--------------------------------
% Description:
% This script processes the direct imaging measurements captured with our 
% ideas coronagraph implementation. In processing these measurements, we 
% run a maximum likelihood estimator for the off-axis exoplanet position at
% various positions in a vertical point source position sweep.
%--------------------------------
% Author(s): Nico Deshler
% Email(s):  ndeshler@arizona.edu
% Date:      July 22, 2024
%--------------------------------

addpath('../Utils/')

%% System Definition
% Constants relevant to the experimental setup.
% system parameters
pinhole_Diameter = 400e-6;  % [units m]
lambda = 532e-9;            % [units m]
focalLength = 200e-3;       % [units m]
sigma = 1.22*lambda*focalLength/pinhole_Diameter; % [units m]

% define coordinates
pixelPitch = 16e-6;                 % [units m]
xx = (-256:255 + .5)*pixelPitch;    % [units m]
yy = (-256:255 + .5)*pixelPitch;    % [units m]
[xx,yy] = meshgrid(xx,yy);          % [units m]

% spatial light modulator
pixelPitch_slm = 8e-6;              % [units m]

% Zernike modes used in measurement
n =[0,1,2,2];                       %  Radial mode indices
m =[0,-1,0,2];                      %  Angular mode indices

% number of images used for star construction
N_star = 1e3; % must be be <= 1000

% relative planet brightness
b = 1/(N_star+1); 

% rayleigh limit
rl = 1.22/2; 

% setup image plane grid in normalized coordinate units
scalefactor = lambda*focalLength/(pinhole_Diameter/2);
X = xx / scalefactor;
Y = yy / scalefactor;


%% Run Maximum Likelihood for Localization and Detection
% Here we will run a maximum likelihood estimator for the exoplanet
% position and the exoplanet hypothesis test on data collected with the 
% ideal coronagraph experimental setup. 
% --------------------------
% EXPERIMENTAL DATA DETAILS
% --------------------------
% Image Dimensions = [N x N]
% Number of off-axis exoplanet locations sampled [K = 80]
% Number of images (trials) collected per off-axis location [L = 100]


% load data 
load('../Data/ExperimentData.mat','exoplanet_imgs','star_imgs','lambda_D','detector_dark_data','xtalk_mtx')
xtalk_mtx = eye(4);

% preprocess the images to turn them into valid measurements
[X, Y, measurement, structured_background] = PreProcessImgs(X, Y, star_imgs, exoplanet_imgs, N_star, lambda_D);

% remove estimated background signal from synthetic measurements
measurement_cleaned = measurement-(N_star+1)*(structured_background + lambda_D);

% suspected exoplanet locations
ynum = 80;
y = rl*pixelPitch_slm/sigma*((1:ynum).'- (ynum+1)/2);        

% estimate magnification (scaling) and pointing (optical axis) inaccuracies
aI = squeeze(sum(mean(measurement_cleaned,4),[1,2])); % integrated intensity
[ff,f0] = BiasScalingFit(n,m,aI/max(aI),y);

% Set true exoplanet locations based on fitted curves (undo bias and scaling)
y_th = y; % store theoretical y (just for reference)               
y = ff.ys*(y-ff.y0);

% run maximum likelihood for position estimate along y axis and hypothesis testing
[y_hat,L, h_hat, P_y, lambda_0] = CoronagraphDirectImagingMLE(X,Y,measurement,xtalk_mtx,N_star,lambda_D,structured_background,n,m,y);

% get throughput curve
pI = squeeze(sum(P_y,[1,2]));

% fit estimators to linear fit and remove DC bias from misalignment
ft_lin = polyfit(repmat(y(abs(y/rl)<.4),[1,size(y_hat,2)]),y_hat(abs(y/rl)<.4,:),1);
y_hat = y_hat - ft_lin(2);

% manual bias correction
y = y - rl*.02;
y_hat = y_hat - rl*.02;

% save measurement model details
save('../Data/MeasurementModel.mat','lambda_0','lambda_D','structured_background','xtalk_mtx','y','y_hat','X','Y')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURES 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display Detector Dark Count Poisson Fit
figure;
h = histogram(detector_dark_data,'FaceColor','k');
hold on
count_rng = h.BinLimits(1):h.BinLimits(2);
poiss_D = poisspdf(count_rng,lambda_D); 
plot(count_rng,poiss_D./max(poiss_D) * max(h.Values),'r','LineWidth',2);
hold off
xlabel('Photodetection Events per Integration Period $T$','interpreter','latex')
ylabel('Frequency','interpreter','latex')
title('Detector Dark Noise','interpreter','latex')
legend({'Dark Detector Data',['Poisson Fit  $\lambda_D =',num2str(lambda_D),'$']},'interpreter','latex')
axis square
grid on
box on
saveas(gcf,'../Figures/SVG/DetectorDarkNoiseHistogram','svg')

%% Display structured background
figure
colormap('turbo')
imagesc([min(X(:)),max(X(:))]/rl,[min(Y(:)),max(Y(:))]/rl,structured_background)
xlabel('$x/\sigma$','interpreter','latex')
ylabel('$y/\sigma$','interpreter','latex')
title('Structured Background','interpreter','latex')
cbar = colorbar;
ylabel(cbar,{'Photodetection Events','per Integration Period $T$'},'interpreter','latex')
axis square
box on
saveas(gcf,'../Figures/SVG/StructuredBackground','svg')
saveas(gcf,'../Figures/FIG/StructuredBackground','fig')

%% Plot integrated intensity as function of position
figure
tiledlayout(1,2)
nexttile(1)
hold on
scatter(y_th/rl,aI/max(aI),10,'filled','k')
plot(y_th/rl,f0,'k','LineWidth',1.5)
hold off
ylabel('Integrated Intensity','interpreter','latex')
xlabel('Exoplanet Position $y_{e}/\sigma, y_{e}^{''}/ \sigma$','interpreter','latex')
legend({'$I^{data}(y_e^{''})$','$I^{theory}(y_e)$'},'interpreter','Latex')
title({'Before Fitting'},'interpreter','latex')
axis square
grid on
box on

nexttile(2)
hold on
scatter(y/rl,aI/max(aI),10,'filled','k')
plot(y/rl,pI/max(pI),'k','LineWidth',1.5)
hold off
ylabel('Integrated Intensity','interpreter','latex')
xlabel('Exoplanet Position $y_{e}/\sigma$','interpreter','latex')
legend({'$I^{data}(y_e)$','$I^{theory}(y_e)$'},'interpreter','Latex')
title({['Bias: $a=', num2str(ff.y0),'$'], ['Scaling: $c=', num2str(ff.ys),'$'], '$y^{''}_{e} = c(y_{e}-a)$','After Fitting'},'interpreter','latex')
axis square
grid on
box on
saveas(gcf,'../Figures/SVG/IntegratedIntensity_BiasScalingFit','svg')
saveas(gcf,'../Figures/FIG/IntegratedIntensity_BiasScalingFit','fig')

%% Log-Likelihood Map
figure;
colormap('turbo')
imagesc([min(y),max(y)]/rl,[min(y),max(y)]/rl,mean(L,3))
axis square; set(gca,'YDir','normal')
hold on
plot(y/rl,y/rl,'k','LineWidth',1)
scatter(y/rl,mean(y_hat,2)/rl,5,'filled','w')
errorbar(y/rl,mean(y_hat,2)/rl,std(y_hat,0,2)/rl,'w')
hold off
xlabel('Exoplanet Position $y_e/\sigma$','interpreter','latex')
ylabel('Estimator Position $\check{y}_e/\sigma$','interpreter','latex')
cbar = colorbar;
ylabel(cbar,{'Normalized Log-Likelihood','$\mathcal{L}(\check{y}_e;y_e)/ |\max_{\check{y}_e} \mathcal{L}(\check{y}_e;y_e)|$'},'interpreter','latex')
title({'Log-Likelihood Map'},'interpreter','latex')
xticks(yticks);
leg = legend({'Perfect Localization','Mean Estimates','$\pm$ Standard Dev.'},'interpreter','latex');
leg.Location = 'SouthEast';
set(leg.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[.5;.5;.5;.3]));
legend('Mean Position Estimates')

saveas(gcf,'../Figures/SVG/LogLikelihoodMap','svg')
saveas(gcf,'../Figures/FIG/LogLikelihoodMap','fig')

%% Estimated Localizations v Ground Truth
figure
hold on
plot(y/rl,y/rl,'k')
legend_names = {'Perfect Estimation','MLE'};

cscale = 7/10;
colormap(hot)
colors = hot(round(size(y_hat,2)/cscale));
for k = 1:length(y)
    [yk,~,ic] = unique(y_hat(k,:));
    hk = hist(ic, numel(yk));
    ck = colors(hk,:);
    scatter(y(k)*ones([1,numel(yk)])/rl,yk/rl,20,ck,'filled')
    legend_names = [legend_names,''];
end
hold off
cbar = colorbar;
cbar.Limits= [0,cscale];
cbar.Ticks = 0:cscale/4:cscale;
cbar.TickLabels = arrayfun(@(j) sprintf('%.f',size(y_hat,2)*j),0:.25:1,'UniformOutput',false);
ylabel(cbar,'Estimate Frequency','interpreter','latex')
xlabel('Exoplanet Position $y_e/\sigma$','interpreter','latex')
ylabel('Estimator Position $\check{y}_e/\sigma$','interpreter','latex')
xticks(yticks);
xlim([-.9,.9])
ylim([-.9,.9])
title({'Exoplanet Localization'},'interpreter','latex')
xline(-.6,'--k','LineWidth',1)
xline(+.6,'--k','LineWidth',1)

leg = legend(legend_names,'interpreter','latex');
leg.Location = 'SouthEast';
axis square
grid on
box on
saveas(gcf,'../Figures/SVG/MLE_Scatter','svg')
saveas(gcf,'../Figures/FIG/MLE_Scatter','fig')

%% Plot Theoretical and Experimental Intensity Distributions at Fixed Frames
sampled_locs = [-.75,-.5,-.25,-.1,.1,.25,.5, .75]; % rayleigh units
[~,id] = min(abs(y/rl - sampled_locs),[],1);

figure
colormap('turbo')
t = tiledlayout(2,numel(sampled_locs),"TileSpacing","compact",'Padding','compact');
for k = 1:numel(sampled_locs)
    % display theoretical intensity profile
    nexttile(k)
    imagesc([min(X(:)),max(X(:))]/rl,[min(Y(:)),max(Y(:))]/rl , (sampled_locs(k)~=0) * P_y(:,:,id(k)));
    hold on; xline(0,'w','LineWidth',1.2); yline(sampled_locs(k),'w','LineWidth',1.2); hold off; axis square; set(gca,'YDir','normal')
    title(sprintf('$y_{e} / \\sigma = %.2f $',sampled_locs(k)),'interpreter','latex'); 
    %xlabel('$x/\sigma$','interpreter','latex'); 
    if k == 1
        ylabel({'THEORETICAL','$y/\sigma$'},'interpreter','latex')
    else
        %ylabel('$y/\sigma$','interpreter','latex')
    end
    xticks([-1,0,1])
    yticks([-1,0,1])
    clim([0,max(P_y(:,:,id),[],'all')])
    
    % display measured intensity profile
    nexttile(k+numel(sampled_locs))
    imagesc([min(X(:)),max(X(:))]/rl,[min(Y(:)),max(Y(:))]/rl, mean(measurement_cleaned(:,:,id(k),:),4));
    psnr_k = psnr(mean(measurement_cleaned(:,:,id(k),:),4)/max(mean(measurement_cleaned(:,:,id(k),:),4),[],'all'), P_y(:,:,id(k)) / max(P_y(:,:,id(k)),[],'all'));
    hold on; 
    xline(0,'w','LineWidth',1.2); yline(sampled_locs(k),'w','LineWidth',1.2); 
    %text(-1.5,-1.3,{'\textbf{PSNR:} ', sprintf('$\\mathbf{%.2f}$',psnr_k)},'Color','m','interpreter','latex');
    hold off; 
    axis square; 
    set(gca,'YDir','normal') 
    xlabel('$x/\sigma$','interpreter','latex');
    xticks([-1,0,1])
    yticks([-1,0,1])
    if k == 1
        ylabel({'EXPERIMENTAL','$y/\sigma$'},'interpreter','latex')
    else
        %ylabel('$y/\sigma$','interpreter','latex')
    end
    clim([0,max(measurement_cleaned(:,:,id,1),[],'all')])
end
saveas(gcf,'../Figures/SVG/FixedFrameMeasurements','svg')
saveas(gcf,'../Figures/FIG/FixedFrameMeasurements','fig')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Truncated Zernike Mode basis

[Xz,Yz] = meshgrid(rl*linspace(-3,3,1001));
[Th,R] = cart2pol(Xz,Yz);
FZ = FourierZernike(R(:),Th(:),n,m); % FZ modes over image plane
FZ = reshape(FZ,[size(Xz), numel(n)]);

colormap('turbo')
tiledlayout(1,numel(n),'TileSpacing','compact','Padding','compact')
for k = 1:numel(n)
    nexttile(k)
    imagesc(abs(FZ(:,:,k)).^2)
    title(sprintf('$|\\psi_{%d}(\\vec{r})|^2$',k-1),'interpreter','latex')
    axis square
    axis off
    box on
end
saveas(gcf,'../Figures/SVG/TruncatedFZModes','svg')
saveas(gcf,'../Figures/FIG/TruncatedFZModes','fig')

%% Generate Video of Measurement with Off-Axis Source Translation
v = VideoWriter('../Figures/VID/ExoplanetScanMLE','MPEG-4');
v.Quality = 100; 
v.FrameRate = 6;
vidfig = figure;
vidfig.Color = [1,1,1];
vidfig.Renderer = 'painters';
vidfig.Position = [50  50  900  900];
open(v)

M = mean(measurement_cleaned,4);
I = sum(measurement_cleaned,[1,2]);
I = I/squeeze(max(mean(I(1,1,:,:),4),[],3));
colors = gray(round(size(y_hat,2)));

t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
colormap('turbo')
for k = 1:size(measurement_cleaned,3)
    
    title(t,{'Exoplanet Position',sprintf('$y_{e} / \\sigma = %.3f $',y(k)/rl)},'interpreter','latex');
    [yk,~,ic] = unique(y_hat(k,:));
    hk = hist(ic, numel(yk));
    hk = min(hk,size(colors,1));
    ck = colors(hk,:);
        
    % display theoretical intensity profile
    nexttile(1)
    imagesc([min(X(:)),max(X(:))]/rl,[min(Y(:)),max(Y(:))]/rl , P_y(:,:,k));
    hold on; xline(0,'w','LineWidth',1.2); yline(y(k)/rl,'w','LineWidth',1.2); hold off; axis square; set(gca,'YDir','normal')
    hold on; scatter(0,0,100,'p','MarkerFaceColor','w','MarkerEdgeColor','k'); hold off;
    
    xlabel('$x/\sigma$','interpreter','latex')
    ylabel('$y/\sigma$','interpreter','latex')
    title({'THEORETICAL'},'interpreter','latex')
    %clim([0,max(P_y(:,:,id),[],'all')])
    
    % display measured intensity profile
    nexttile(2)
    imagesc([min(X(:)),max(X(:))]/rl,[min(Y(:)),max(Y(:))]/rl ,M(:,:,k));
    hold on; xline(0,'w','LineWidth',1.2); yline(y(k)/rl,'w','LineWidth',1.2); hold off; axis square; set(gca,'YDir','normal') 
    xlabel('$x/\sigma$','interpreter','latex')
    ylabel('$y/\sigma$','interpreter','latex')
    title({'EXPERIMENTAL'},'interpreter','latex')
    hold on; scatter(zeros(size(yk)),yk/rl,30,'MarkerFaceColor','w','MarkerEdgeColor','k'); hold off;
    hold on; scatter(0,0,100,'p','MarkerFaceColor','w','MarkerEdgeColor','k'); hold off;
    %clim([0,max(measurement_cleaned(:,:,id,1),[],'all')])


    % display theoretical and integrated intensities
    nexttile(3)
    
    hold on
    plot(y(1:k)'/rl,squeeze(sum(P_y(:,:,1:k),[1,2])/max(sum(P_y,[1,2]),[],3)),'k')
    scatter(y(1:k)'/rl,squeeze(mean(I(1,1,1:k,:),4)),'r')
    errorbar(y(1:k)'/rl,squeeze(mean(I(1,1,1:k,:),4)),squeeze(std(I(1,1,1:k,:),0,4)),'r')
    hold off
    xlim([-1.1,1.1])
    ylim([0,1.1])
    ylabel('Normalized Energy','interpreter','latex')
    xlabel('Exoplanet Position $y_e/\sigma$','interpreter','latex')
    axis square
    leg = legend({'Theoretical','Experimental'},'interpreter','latex');
    leg.Location = 'North';
    title('THROUGHPUT')

    % display estimated point spread
    nexttile(4)
    hold on
    plot(y/rl,y/rl,'k')
    legend_names = {'Perfect Localization','MLEs'};
    cscale = 5*7/10;
    colors = hot(round(size(y_hat,2)/cscale));
    for j = 1:k
        [yk,~,ic] = unique(y_hat(j,:));
        hk = hist(ic, numel(yk));
        hk = min(hk,size(colors,1));
        ck = colors(hk,:);
        scatter(y(j)*ones([1,numel(yk)])/rl,yk/rl,20,ck,'filled')
        legend_names = [legend_names,''];
    end
    hold off
    %cbar = colorbar;
    %cbar.Limits= [0,cscale];
    %cbar.Ticks = 0:cscale/4:cscale;
    %cbar.TickLabels = arrayfun(@(j) sprintf('%.f',size(y_hat,2)*j),0:.25:1,'UniformOutput',false);
    %ylabel(cbar,'Estimate Frequency','interpreter','latex')
    xlabel('Ground Truth Position $y_e/\sigma$','interpreter','latex')
    ylabel('Estimator Position $\check{y}_ed/\sigma$','interpreter','latex')
    xticks(yticks);
    axis square
    leg = legend(legend_names,'interpreter','latex');
    leg.Position = [0.6871    0.0871    0.2103    0.0698];
    title('MAX LIKELIHOOD ESTIMATES')

    set(gca,'color','w')
    frame = getframe(vidfig);
    writeVideo(v,frame)
end
close(v)

%% Functions
function [X,Y, measurement,structured_background] = PreProcessImgs(X,Y,star_imgs, exo_imgs, N_star, lambda_D)
    % Here we add the star and planet images to faithfully represent two
    % incoherent sources in the measurement
    
    %%%% SHAPING %%%
    % crop images
    x0=396;
    y0=145;
    d=76; % must be even

    exo_imgs = exo_imgs((x0-d/2):(x0+d/2),(y0-d/2):(y0+d/2),:,:);
    star_imgs = star_imgs((x0-d/2):(x0+d/2),(y0-d/2):(y0+d/2),:);
    
    % transpose images
    exo_imgs = pagetranspose(exo_imgs);
    star_imgs = pagetranspose(star_imgs);
    
    % crop supports to X and Y to match
    dx = d; dy = d;
    X = X(((size(X,1)-dy)/2 + 1):((size(X,1)+dy)/2 + 1),((size(X,2)-dx)/2 + 1):((size(X,2)+dx)/2 + 1));
    Y = Y(((size(Y,1)-dy)/2 + 1):((size(Y,1)+dy)/2 + 1),((size(Y,2)-dx)/2 + 1):((size(Y,2)+dx)/2 + 1));
    %%%%%%%%%%%%%%%%
    
    % make star image
    star_img = sum(star_imgs(:,:,1:N_star),3);

    % add star and planet images
    measurement = star_img + exo_imgs;

    % estimate structured background
    structured_background = mean(measurement(:,:,39:40,:),[3,4])/(N_star+1) - lambda_D;

    %% add poisson-like noise to measurements (more valid)
    measurement = measurement + randn(size(measurement)) .* sqrt((N_star + 1)*structured_background);
end



function [y_hat, L, h_hat, P_y, lambda_0] =  CoronagraphDirectImagingMLE(X,Y,measurement,xtalk_mtx,N_star,lambda_D,structured_background,n,m,y)
    % computes the maximum likelihood location and hypothesis test for
    % estimating the position and heralding the presence/absence of an
    % exoplanet.
    %
    % ---------------------------------------------------------------------
    % ------------------------------- INPUTS ------------------------------
    % ---------------------------------------------------------------------
    % X,Y:          [NxN] coordinate grids 
    % measurement:  [NxNxK1xT] array of intensity measurements 
    % xtalk_mtx:    [Q,Q] crosstalk matrix (generally complex)
    % N_star:       [1] number of images added to create the star (integer)
    % lambda_D:     [1] background dark click rate
    % structured_background:    [N,N] structured background rate image
    % n,m:          [1xQ] zernike mode indices
    % y:            [K1,1] range of y-values sampled in the data
    % ---------------------------------------------------------------------
    % ------------------------------ OUTPUTS ------------------------------
    % ---------------------------------------------------------------------
    % y_hat:      [K1,T] estimated exoplanet locations
    % L:          [K1,K2] max-normalized log likelihood map
    % h_hat:      [K1,T] hypothesis test declaration
    % P_y:        [N,N,M] idealized intensity distributions.


    % constants
    rl = 1.22/2;            % rayleigh constant
    b = 1/(N_star + 1);     % relative brightness of planet to star
    d2x = (X(1,2)-X(1,1))^2;% differential area element
    
    % reshape the measurement array to [N^2,1,K1,T]
    dm = size(measurement);
    measurement = reshape(measurement,[dm(1)*dm(2),1,dm(3),dm(4)]);

    % reshape structured background
    structured_background = structured_background(:);
    
    % get probability distribution over image plane for each possible source location
    [Th,R] = cart2pol(X,Y);
    FZ = FourierZernike(R(:),Th(:),n,m); % FZ modes over image plane

    % upsample the ground truth y locations for more refined MLE search space
    upsample_factor = 5;
    temp = linspace(min(y),max(y),upsample_factor*numel(y)).';
    [~,y_id] = min(abs(y-temp.'),[],2);
    y = temp;                   % [K2,1]

    % zernike expansion coeffs and the attenuation coeffs
    [th_y,r_y] = cart2pol(zeros(size(y)),y);
    zs = conj(FourierZernike(0,0,n,m)).'/sqrt(pi);       % star expansion coeffs
    ze = conj(FourierZernike(r_y,th_y,n,m)).'/sqrt(pi);  % planet expansion coeffs
    attenuation_coeffs = (n~=0 | m~=0).';

    % Transform zernike coeffs as if they were propagating through system
    % crosstalk on forward pass
    ze = xtalk_mtx' * (attenuation_coeffs .* (xtalk_mtx * ze));
    zs = xtalk_mtx' * (attenuation_coeffs .* (xtalk_mtx * zs));                                          
    % reshape
    zs = permute(zs,[3,1,2]);
    ze = permute(ze,[3,1,2]);                   % [1,Q,ynum]
    
    % probability distribution of intensity over image-plane for off-axis
    % planet in presence of star
    qs = abs(sum(zs .* FZ, 2)).^2;
    qe = abs(sum(ze .* FZ, 2)).^2;
    P_y = d2x*((1-b) * qs + b * qe);    % [N^2,1,K2]
    P_y = squeeze(P_y);                 % [N^2,K2]

    % estimate mean photon rate at pupil (intensity per measurement integration time) 
    lambda_0_i = (mean(measurement,4)/(N_star + 1) - structured_background - lambda_D)./permute(P_y(:,y_id),[1,3,2]);
    mask = (lambda_0_i ~= 0) & isfinite(lambda_0_i) & (lambda_0_i >=0);
    lambda_0_i = lambda_0_i(mask);
    
    % plot histogram of candidate photon rates
    figure
    hg = histogram(log10(lambda_0_i),'FaceColor','k');
    xlabel('$\log_{10}(\lambda_0)$','interpreter','latex')
    ylabel('Number of Ocurrences','interpreter','latex')
    title('$\lambda_0$ Estimates','interpreter','latex')
    axis square
    grid on
    [~,id] = max(hg.Values);
    lambda_0 = 10^((hg.BinEdges(id)+hg.BinEdges(id+1))/2);
    hold on
    xline(log10(lambda_0),'--r','LineWidth',2)
    hold off
    legend({'Estimates $\check{\lambda}_0^{(i)}$','Best Linear Unbiased Est. $\check{\lambda}_0$'},'interpreter','latex')
    
    % Best linear unbiased estimator (BLUE) - https://en.wikipedia.org/wiki/Weighted_least_squares
    residual = @(x) (mean(measurement,4)/(N_star + 1) - structured_background - lambda_D - x*permute(P_y(:,y_id),[1,3,2]));
    variances = @(x) (structured_background + lambda_D + x*permute(P_y(:,y_id),[1,3,2]));
    S = @(x) sum(residual(x).^2./variances(x),'all');
    lambda_0 = fminsearch(S,lambda_0);
    lambda_0 = lambda_0*1.05;

    %% EXOPLANET LOCALIZATION
    % poisson -> gaussian rate
    mu = (N_star+1) * ( lambda_0 * P_y + structured_background + lambda_D);

    % get log-likelihood for all measurements
    LL = zeros(numel(y),dm(3),dm(4));
    for k = 1:dm(4)
         
        % log-likelihood
        ll = -(measurement(:,:,:,k) - mu).^2 ./ mu / 2 - .5* log(2*pi*mu);

        % clean up any invalid values before summing
        ll(ll==-inf | isnan(ll)) = 0;

        % sum likelihoods over all pixels
        LL(:,:,k) = squeeze(sum(ll,1)); % [K2,K1,T]
        
    end
    
    % get max along ynum dimension
    L = LL ./ abs(max(LL,[],1));
    [~,V] = sort(L,1);
    y_hat = squeeze(y(V(end,:,:)));


    %% EXOPLANET DETECTION
    % given estimated exoplanet location, determine whether exoplanet is
    % present or not.
    
    % compute intensity probability distributions over detector for estimated exoplanet locations 
    [th_y,r_y] = cart2pol(zeros(size(y_hat)),y_hat);
    ze = conj(FourierZernike(r_y(:),th_y(:),n,m)).'/sqrt(pi);  % planet expansion coeffs
    
    % propagate zernike expansion coeffs through coronagraph
    ze = xtalk_mtx' * (attenuation_coeffs .* (xtalk_mtx * ze));                                                
    ze = permute(ze,[3,1,2]);                   % [1,Q,ynum]
    qe = abs(sum(ze .* FZ, 2)).^2;
    qe = reshape(qe,[dm(1)*dm(2),size(y_hat)]);


    % probability distributions over image plane for either hypothesis
    P0 = d2x*qs;
    P1 = d2x*((1-b) * qs + b * qe);

    % poisson -> gaussian rates for either hypothesis
    mu0 = (N_star+1) * ( lambda_0 * P0 + structured_background + lambda_D);
    mu1 = (N_star+1) * ( lambda_0 * P1 + structured_background + lambda_D);

    % log likelihoods for either hypothesis
    LL0 = - (squeeze(measurement) - mu0).^2 ./ mu0 / 2 - .5* log(2*pi*mu0); % null hypothesis (no planet)
    LL1 = - (squeeze(measurement) - mu1).^2 ./ mu1 / 2 - .5* log(2*pi*mu1); % alt hypothesis (planet)

    % clean up any invalid values before summing
    LL0(LL0==-inf | isnan(LL0)) = 0;
    LL1(LL1==-inf | isnan(LL1)) = 0;
    
    % sum likelihoods over all pixels
    LL0 = squeeze(sum(LL0,1));
    LL1 = squeeze(sum(LL1,1));
    
    % hypothesis test
    h_hat = LL1>LL0;


    %% Reshape certain outputs
    P_y = reshape(P_y(:,y_id),[dm(1:2),numel(y_id)]);
end



function [ff,f0] = BiasScalingFit(n,m,throughput,y)
    % determines the scaling and bias of the y-axis due to experimental
    % misalignment from the throughput curves
    
    % fit theoretical curve to throughput
    function f = helper(y0,ys,y,n,m)
        attenuation_coeffs = (n~=0 | m~=0);
        [th_y, r_y] = cart2pol(zeros(size(y)),ys*(y-y0));
        f = sum(abs(attenuation_coeffs.*conj(FourierZernike(r_y,th_y,n,m))/sqrt(pi)).^2,2);
        f = f./max(f);
    end

    f0 = helper(0,1,y,n,m);% original function before fit

    lambda = @(y0,ys,x) helper(y0,ys,x,n,m);
    ft = fittype(lambda);
    ff = fit(y,throughput,ft,'StartPoint',[0,1]);    
end