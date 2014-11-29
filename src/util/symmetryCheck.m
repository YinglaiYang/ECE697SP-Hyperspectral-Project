function symmetric =  symmetryCheck( variable )
%SYMMETRYCHECK Summary of this function goes here
%   Detailed explanation goes here
symmetric = issymmetric(variable);

fprintf('Is %s symmetric?: %d\n', inputname(1), symmetric);

end

