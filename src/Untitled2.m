newMass = zeros(1,19);

for l=1:19
    newMass(l) = nnz(predictedLabels == l);
end