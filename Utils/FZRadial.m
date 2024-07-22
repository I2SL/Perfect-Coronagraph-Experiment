function u = FZRadial(r,n)
    % Computes the radial function of the Fourier Transformed Zernikes
    nn = repmat(n,[size(r,1),1,size(r,3)]);
    rr = repmat(r,[1,size(n,2),1]);

    % sinc-bessel in polar
    J = besselj(nn+1,2*pi*rr) ./ (sqrt(pi) * rr);
    
    % fill in singularities
    J(r==0 , n+1 == 1) = sqrt(pi);
    J(r==0 , n+1 > 1) = 0;
    
    % radial function
    u = J;
end