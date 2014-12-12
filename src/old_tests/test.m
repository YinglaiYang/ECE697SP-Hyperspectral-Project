load mapTest;

m_F_star = max(F_star, [], 2);

mapPredict = zeros(n,1);

for k=1:n
    mapPredict(k) = find(F_star(k,:) == m_F_star(k));
end

mapPredict = reshape(mapPredict, size(mapTrain));

testIDX = mapTest > 0;

N_test = nnz(mapTest);

testError = nnz(mapPredict(testIDX) ~= mapTest(testIDX)) / N_test;