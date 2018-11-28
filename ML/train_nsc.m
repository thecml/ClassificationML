function centroids = train_nsc(trainData, trainLbls, nClasses, nSubClasses) 

%preprocessor
%sort the train data samples and labels in ascending order.
nPixels = size(trainData,1);
trainData = sortrows([trainData; trainLbls']',nPixels+1);
trainLbls = sortrows(trainLbls);
trainData = trainData(:,1:nPixels)';

%kmeans for nSubClasses
%column(i:i+1) belongs to class K
classes = trainLbls(1):nClasses
centroids = zeros(size(trainData,1), nClasses*nSubClasses);
n = zeros(size(classes,1),size(classes,2))';
nIndex = 1;
centroidIdx = 1:nSubClasses:nClasses*nSubClasses;

for i = 1:nClasses
    n(i,:) = sum(trainLbls==classes(i));
    [idx,clst] = kmeans(trainData(:,nIndex:nIndex+n(i))', nSubClasses);
    nIndex = nIndex+n(i)-1;
    centroids(:,centroidIdx(i):centroidIdx(i)+1) = clst';
end
end