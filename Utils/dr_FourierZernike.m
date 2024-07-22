function dr_FZ = dr_FourierZernike(r,th,n,m)
    % Computes the partial derivative of the Fourier Zernike wrt the radial
    % coordinate.
    nn = repmat(n,[size(r,1),1,size(r,3)]);
    rr = repmat(r,[1,size(n,2),1]);

    dr_FZ = (-1).^(n/2 + abs(m)) .* sqrt(pi./(n+1)).*(besselj(nn-1,2*pi*rr)-besselj(nn+3,2*pi*rr)).*FZAngle(th,m);
end