function [n,m] = ZernikeIndices(nmax)
    %--------------------------------
    % Description: 
    % returns all Zernike indices up to radial index nmax using OSA/ANSI
    % indexing standard.
    %--------------------------------
    % Author(s): Nico Deshler
    % Email(s):  ndeshler@arizona.edu
    % Date:      July 22, 2024
    %--------------------------------

    N = (nmax+1)*(nmax+2)/2;
    n = zeros(1,N);
    m = zeros(1,N);
    
    k = 1;
    for nn = 0:nmax
        for mm = -nn:2:nn
            n(k) = nn;
            m(k) = mm;
            k = k+1;
        end
    end
end