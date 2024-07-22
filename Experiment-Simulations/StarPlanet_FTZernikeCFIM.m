function [CFI_nm_rr, CFI_nm_th, CFI_nm_xt,Pe_nm,Ps_nm,P_nm] = StarPlanet_FTZernikeCFIM(r_delta,th_delta,n,m,b)
%--------------------------------
% Description: 
% Returns the CFI matrix elements for the separation vector between an 
% intensity-centroid-aligned star-planet system when imaging with an ideal 
% coronagraph that rejects the fundamental mode of a circular telescope. 
%--------------------------------
% Author(s): Nico Deshler
% Email(s):  ndeshler@arizona.edu
% Date:      July 22, 2024
%--------------------------------

%---- INPUTS -----
% r_delta  : [Nx1]
% th_delta : [Nx1]
% n        : [1xM]
% m        : [1xM]
% b        : K

%---- OUTPUTS -----
% CFI_nm_rr: [N x M x K]    CFI of radial parameter
% CFI_nm_th: [N x M x K]    CFI of angular parameter
% CFI_nm_xt: [N x M x K]    CFIM cross terms
% Pe_nm    : [N x M x K]    Joint probability of photon in mode nm and emitted by exoplanet
% Ps_nm    : [N x M x K]    Joint probability of photon in mode nm and emitted by star
% P_nm     : [N x M x K]    Total mode probability


N = numel(r_delta);
M = numel(n);
K = numel(b);

% helper functions
bsj = @(r,n) besselj(repmat(n,[size(r,1),1]), repmat(r,[1,size(n,2)]));
p_nm = @(r,th,n,m) abs(FourierZernike(r,th,n,m)).^2 / pi;
dp_nm_rr = @(r,th,n,m) 2 * bsj(2*pi*r,n+1) ./ repmat(r,[1,size(n,2)]) .* ( bsj(2*pi*r,n-1) - bsj(2*pi*r,n+3) ) .* (FZAngle(th,m)).^2;
dp_nm_th = @(r,th,n,m) 4 * (bsj(2*pi*r,n+1) ./ (pi * repmat(r,[1,size(n,2)]))).^2 .* (n+1) .* abs(m) .* cos(abs(m).*th) .* sin(abs(m).*th) .* sign(-m);

% instantiate outputs
Pe_nm = zeros([N,M,K]);
Ps_nm = zeros([N,M,K]);
P_nm = zeros([N,M,K]);
CFI_nm_rr = zeros([N,M,K]);
CFI_nm_th = zeros([N,M,K]);
CFI_nm_xt = zeros([N,M,K]);

for k = 1:numel(b)
    
    % star and planet coordinates
    r_s = b(k) * r_delta;
    th_s = rem(th_delta + pi,2*pi);
    r_e = (1-b(k)) * r_delta;
    th_e = th_delta;
    
    % radial derivative of mode probabilities at star and planet locations
    dp_rr_s = dp_nm_rr(r_s,th_s,n,m);
    dp_rr_e = dp_nm_rr(r_e,th_e,n,m);

    % total radial derivative
    dP_nm_rr = b(k).*(1-b(k)).*(dp_rr_s + dp_rr_e);

    % azimuthal derivative of mode probabilities at star and planet
    % locations
    dp_th_s = dp_nm_th(r_s,th_s,n,m);
    dp_th_e = dp_nm_th(r_e,th_e,n,m);
    
    % total azimuthal derivative
    dP_nm_th = (1-b(k))*dp_th_s + b(k)*dp_th_e;
    
    % P_star
    Ps_nm(:,:,k) = (1-b(k)).*p_nm(r_s,th_s,n,m);
    
    % P_exoplanet
    Pe_nm(:,:,k) = b(k).*p_nm(r_e,th_e,n,m);
    
    % total probability
    P_nm(:,:,k) = Ps_nm(:,:,k) + Pe_nm(:,:,k);
    
    % CFI radial component (CFIM(1,1))
    CFI_nm_rr(:,:,k) = (dP_nm_rr).^2 ./ (P_nm(:,:,k) + realmin);
    
    % CFI azimuthal component (CFIM(2,2))
    CFI_nm_th(:,:,k) = (dP_nm_th).^2 ./ (P_nm(:,:,k) + realmin);
    
    % CFI cross-term
    CFI_nm_xt(:,:,k) = (dP_nm_rr.*dP_nm_th)./ (P_nm(:,:,k) + realmin);
end

end
