clear, clc, close all;

runs = 1e2;

spectral_norm = false; %if false use Frobenius instead - this is faster than L2-norm/spectral norm

set_self_affinity_to_zero = false;

kMeans = false; % k-means algorithm should not be tested with this script! 
                % There are no cluster in the data (`x`) for this case!
                % It is uniformly distributed!

regulization_param = 0;

n = 100;
m = 30;

sigma = 0.1;

p_array = ceil([0.01 0.05 0.075 0.1 0.15 0.25 0.5 0.75 1] * m);

avg_error_matrix_norm = zeros(1, length(p_array)); max_error_norm = zeros(1, length(p_array));


runtime = tic;

for q=1:length(p_array)
    p = p_array(q);
    
    for r=1:runs
        disp(num2str(r));

        x = rand(n,1);

        S = getRBF_AffinityMatrix(x, 1:n, sigma);

        if set_self_affinity_to_zero
            S = S - diag(diag(S));
        end
            
        if ~kMeans          
            mIDX = randsample(n, m);

        %     [V, D] = eigs(S, m); S_eigs = V*D*V.';

            [V_mp, Lambda_pp] = eig(S(mIDX,mIDX));

            [~,idx] = sort(diag(Lambda_pp),1,'descend');
            V_mp = V_mp(:, idx);
            d_vec = diag(Lambda_pp);
            Lambda_pp = diag(d_vec(idx));

            V_mp = V_mp(:,1:p);
            Lambda_pp = Lambda_pp(1:p,1:p);

            V_tilde = sqrt(m/n) * S(:,mIDX) * (V_mp * Lambda_pp^-1);    
            Lambda_tilde = (n/m) * Lambda_pp;                

            S_tilde = V_tilde * Lambda_tilde * V_tilde.';
        else
            kernel = struct('type', 'rbf', 'para', sigma^2);
            
            S_tilde = INys(kernel,x, m, 'k');
        end
%         memory

    %     S_tilde2 = S_nm * S_mm^-1 * S_nm.';

    %     norm(S-S_tilde)
    %     norm(S-S_tilde2)

        % Diff = S - S_tilde;

        if spectral_norm
            error_norm = norm(S - S_tilde);
        else
            error_norm = norm(S - S_tilde, 'fro');
        end

        avg_error_matrix_norm(q) = avg_error_matrix_norm(q) + error_norm;

        if error_norm > max_error_norm(q)
            max_error_norm(q) = error_norm;
        end
    end
end

avg_error_matrix_norm = avg_error_matrix_norm / runs;

toc(runtime)

figure, plot(p_array, avg_error_matrix_norm);