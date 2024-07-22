%% Mode Intensities
%--------------------------------
% Description: 
% This script segments intensity measurements made at the sorting plane of 
% the experimental coronagraph over the course of a point source translation
% sweep in order to characterize the crosstalk matrix. NOTE: Sorting plane
% mode segmentation masks are somewhat heuristically chosen. Marginal light
% leakage into residual regions of the sorting plane are not accounted for.
%--------------------------------
% Author(s): Nico Deshler
% Email(s):  ndeshler@arizona.edu
% Date:      July 22, 2024
%--------------------------------

% load images of sorter plane
load('../Data/SorterPlaneImages.mat')

% point source scan positions
y = linspace(-0.5547,0.4930,86);

% rayleigh limit
rl = 1.22/2;

% coordinate grid
[X,Y] = meshgrid(linspace(-1,1,512));

% mask radii
R = 1/sqrt(200);

% center of each mode spot
v0 = [-.33,+.6];    c0 = ((X-v0(1)).^2 + (Y-v0(2)).^2 <= R^2);
v1 = [-.16,+.62];   c1 = ((X-v1(1)).^2 + (Y-v1(2)).^2 <= R^2);
v2 = [-.34,+.44];   c2 = ((X-v2(1)).^2 + (Y-v2(2)).^2 <= R^2);
v3 = [-.15,+.44];   c3 = ((X-v3(1)).^2 + (Y-v3(2)).^2 <= R^2);

% segmentation
Segmentation = cat(3,c0,c1,c2,c3);

% crop images
d1 = 340:440; d2 = 145:245;
SorterPlaneImages = SorterPlaneImages(d1,d2,:,:);
Segmentation = Segmentation(d1,d2,:);
X = X(d1,d2); Y=Y(d1,d2);

% Mode Intensities
M = squeeze(sum(SorterPlaneImages .* Segmentation,[1,2]));

% Scale mode intensities (normalize)
M = M-min(M,[],2);
M = M./sum(M,1);

% save mode intensity data
save('../Data/ModeIntensities.mat','M','y')

%% FIGURES
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');


%% DISCRETE SAMPLE OF CALIBRATION IMAGE SCANS
c = [1,1,1;
    1,0,0;
    0,1,0;
    0,.5,1];

centers = [v0;v1;v2;v3];
radii = ones(4,1)*R;

figure
ntile = 5;
colormap(slanCM('magma'))
tiledlayout(1,ntile,"TileSpacing","compact","Padding","compact")
i_samp = round(linspace(1,86,ntile));
y_samp = rl*linspace(-.85,.85,ntile);
for i = 1:ntile
    nexttile(i)
    imagesc([min(X(:)),max(X(:))],[min(Y(:)),max(Y(:))],SorterPlaneImages(:,:,1,i_samp(i)))
    title(['$y/\sigma=',sprintf('%.2f',y_samp(i)/rl),'$'])
    axis off
    axis square
    hold on
    for k =1:4
        h = viscircles(centers(k,:),radii(k),'Color',c(k,:),'LineWidth',1);
        h.Children(2).LineWidth=1e-9;
        h.Children(2).Color=h.Children(1).Color;
        
        
    end
    hold off
    %set(gca,'YDir','normal')
end
saveas(gcf,'../Figures/FIG/SortingPlaneMeasurements','fig')
saveas(gcf,'../Figures/SVG/SortingPlaneMeasurements','svg')

%% VIDEO OF MODE INTENSITY SCAN
figure
tiledlayout(1,2)
v = VideoWriter('../Figures/VID/ModeIntensityScan','MPEG-4');
v.open
colormap(slanCM('magma'));

for i = 1:86
    nexttile(1)
    imagesc(SorterPlaneImages(:,:,:,i))
    title('Sorting Plane Raw','interpreter','latex')
    axis square
    axis off
    
    nexttile(2)
    imagesc(SorterPlaneImages(:,:,:,i).*sum(Segmentation,3))
    title('Sorting Plane Segmented','interpreter','latex')
    axis square
    axis off

    frame = getframe(gcf);
    writeVideo(v,frame)
end
close(v)



