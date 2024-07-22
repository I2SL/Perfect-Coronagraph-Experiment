function z = FourierZernike(r,theta,n,m)
%--------------------------------
% Description: 
% Returns the evaluation of Fourier Transform of the Zernike Polynomials
% function.
%--------------------------------
% Author(s): Nico Deshler
% Email(s):  ndeshler@arizona.edu
% Date:      July 22, 2024
%--------------------------------
%
%---- INPUTS -----
% n - radial index          (row vector 1xK)
% m - azimuthal index       (row vector 1xK)
% r - radial argument       (col vector Dx1)
% theta - angular argument  (col vector Dx1)
% (r,theta) are coordinate pairs, (n,m) are index pairs
%
%---- OUTPUTS -----
% z - the function output   (matrix DxK)
  
z = (-1).^(n/2 + abs(m)) .* sqrt(n+1) .* FZRadial(r,n) .* FZAngle(theta,m);    
end