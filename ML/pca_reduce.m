function pc = pca_reduce(trainData, n_dimensions)

%remove the mean value variable-wise (row-wise), centering
trainData=trainData-repmat(mean(trainData,2) ...
    ,1,size(trainData,2));

%calculate eigenvectors W, and eigenvalues of the covar matrix
[W, EvalueMatrix] = eig(cov(trainData'));

%order by largest eigenvalue
W = W(:,end:-1:1);

%generate PCA component space (PCA scores)
pc = W(:,1:n_dimensions)'*trainData;

%plot PCA space of the first two PCs: PC1 and PC2
%figure
%hold on
%scatter(pc(1,:),pc(2,:),[], trainLbls)

%hold on
%scatter(1:length(resLabels),resLabels, [], 'red')
%scatter(1:length(testLbls),testLbls, [], 'blue')
end
