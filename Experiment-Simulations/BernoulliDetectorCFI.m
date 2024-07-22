%% Bernoulli Detector Classical Fisher Information
%--------------------------------
% Description: 
% Computes the CFI under a Bernoulli noise process for each pixel at 
% the detector used in a hypothetical optimal coronagraphy experiment. 
% The goal of this script is to characterize how a random detector noise
% process affects the information content of each photoelectron registered
% in the context of exoplanet imaging. The probability of detection at each
% pixel is characterized by the mean intensity profile plus a bernoulli 
% random variable (either 1 or 0) at each pixel with probability q.
%--------------------------------
% Author(s): Nico Deshler
% Email(s):  ndeshler@arizona.edu
% Date:      July 22, 2024
%--------------------------------

addpath('../Utils/')

% Star-planet separations
rl = 1.22/2;                            % rayleigh length
r_delta = rl*10.^linspace(-5,1,300);    % star-planet separation

% image plane
num_rl = 102;
ndim = 520;
x = num_rl * rl * linspace(-.5,.5,ndim);
d2x = (x(2)-x(1))^2;
[X,Y] = meshgrid(x);                    % Cartesian coords
[T,R] = cart2pol(X,Y);                  % Polar coords

% helper functions
bsj = @(r,n) besselj(repmat(n,[size(r,1),1]), repmat(r,[1,size(n,2)]));
dPsi0 = @(r,th) pi^(3/2) * ( bsj(2*pi*r,-1) - bsj(2*pi*r,3)) .* FZAngle(th,0);

% Bernoulli Probabilities
q = [0;10.^-fliplr(1:5)'];

% Populate detection probabilities and derivatives
P = zeros([numel(X),numel(r_delta),numel(q)]);
dP = zeros([numel(X),numel(r_delta)]); 

for k = 1:numel(r_delta)
    % coordinates shifted to exoplanet position
    [te,re] = cart2pol(X(:) - r_delta(k), Y(:));

    % direct imaging photon arrival probability over pixels due to
    % exoplanet at position r_delta(k) plus additive bernoulli random
    % process.
    P(:,k,:) = abs(FourierZernike(re,te,0,0) - 1/sqrt(pi) * FourierZernike(r_delta(k),0,0,0) .* FourierZernike(R(:),T(:),0,0) ).^2 ...
                + permute(q,[3,2,1]);

    % derivative of direct imaging probability w.r.t exoplanet position
    dP(:,k) = 2*(FourierZernike(re,te,0,0) - 1/sqrt(pi) * FourierZernike(r_delta(k),0,0,0) .* FourierZernike(R(:),T(:),0,0))...
                               .*(dPsi0(re,-te).*(r_delta(k) - R(:).*cos(T(:)))./re ...
                               - 1/sqrt(pi) * dPsi0(r_delta(k),0) .* FourierZernike(R(:),T(:),0,0));
end

% CFI for direct imaging coronagraph with bernoulli noise at detector
CFI = (dP).^2 ./ P;
CFI = reshape(CFI,[ndim,ndim,numel(r_delta),numel(q)]);
CFI_curve = squeeze(d2x*sum(CFI,[1,2]));

% Quantum Fisher Information (shot-noise limited)
QFI = 4*pi^2*(1 - ( 2 * besselj(2,2*pi*r_delta)./(pi * r_delta)).^2);

%% Plot CFI Performance
figure
hold on
plot(r_delta/rl,QFI/4/pi^2,'k','LineWidth',1.5) % QFI
plot(r_delta/rl,CFI_curve/4/pi^2,'LineWidth',1.5) % CFI
hold off
xlabel('Source Location $y_e/\sigma$','interpreter','latex')
set(gca,'xscale','log')
ylabel('Coronagraph Fisher Information  $/ 4 \pi^2$','interpreter','latex')
title({'Classical Fisher Information','Under Different Bernoulli Backgrounds for Single-Photon States'},'interpreter','latex')
leg = legend([{'QFI',},arrayfun(@(j)sprintf('%.e',q(j)),1:numel(q),'UniformOutput',false)]);
leg.Location = 'West';
title(leg,'Bernoulli Prob $p_B$','interpreter','latex')
ylim([0,1])
xlim([min(r_delta),max(r_delta)]/rl)
axis square

saveas(gcf,'../Figures/SVG/BernoulliNoise','svg')
saveas(gcf,'../Figures/FIG/BernoulliNoise','fig')
