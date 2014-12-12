%% Following parameters are used for crossvalidation
%% $\alpha$
alphas = [0.01, 0.1, 0.25, 0.5, 0.75, 0.99];

%% $\sigma$
sigmas = [1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1];

%% save parameters
save('alphas.mat', 'alphas');
save('sigmas.mat', 'sigmas');

%% Nyström parameters
nystroemFraction = 1e-3; % 1percent

RSVD.k_fraction = 1.0;
RSVD.p = 10;
RSVD.q = 3;