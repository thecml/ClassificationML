function centroids = train_nsc(trainData, trainLbls, nClasses, nSubClasses) 

%Nearest Subclass Centroid
%kmeans for nSubClasses
%column(i:i+1) belongs to class K
classes = trainLbls(1):nClasses
centroids = zeros(size(trainData,1), nClasses*nSubClasses);
n = zeros(size(classes,1),size(classes,2))';
nIndex = 1;
centroidIdx = 1:nSubClasses:nClasses*nSubClasses;

for i = 1:nClasses
    n(i,:) = sum(trainLbls==classes(i));
    [idx,clst] = kmeans(trainData(:,nIndex:nIndex+n(i)-1)', nSubClasses);
    nIndex = nIndex+n(i);
    centroids(:,centroidIdx(i):centroidIdx(i)+nSubClasses-1) = clst';
end
end