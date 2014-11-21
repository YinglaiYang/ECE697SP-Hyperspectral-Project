classdef NystroemUniform < handle
    %NYSTROEMUNIFORM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = private)
        sampledIndices
    end
    
    methods
        function NU = NystroemUniform(n, m)
            NU.sampledIndices = randsample(n, m);
        end
    end
    
end

