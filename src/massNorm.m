
mass = zeros(1,19);
mass(1) = 5;
mass(2) = 1000;
mass(3) = 1000;
mass(4) = 6;
mass(5) = 55;
mass(6) = 1000;
mass(7) = 5;
mass(8) = 41;
mass(9) = 545;
mass(10) = 21;
mass(11) = 1000;
mass(12) = 608;
mass(13) = 633;
mass(14) = 1000;
mass(15) = 1000;
mass(16) = 51;
mass(17) = 53;
mass(18) = 5;
mass(19) = 1000;

normFactor = 1./mass;

F_star2 = F_star.*repmat(normFactor, size(F_star,1), 1);