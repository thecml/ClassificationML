function mu = train_nc(trainData, trainLbls, nClasses) 

%preprocessor
%sort the samples and labels in ascending order.
nPixels = size(trainData,1);
trainData = sortrows([trainData; trainLbls']',nPixels+1);
trainLbls = sortrows(trainLbls);
trainData = trainData(:,1:nPixels)';

%n = num of samples in each training class
%find a matrix with the means of the classes using sorted data
mu = zeros(size(trainData,1), nClasses);
classes = 0:nClasses-1;
n = zeros(size(classes,1),size(classes,2))'
nIndex = 1;

for i = 1:nClasses
    n(i,:) = sum(trainLbls==classes(i));
    mu(:,i) = sum(trainData(:,nIndex:nIndex+n(i))')/n(i);
    nIndex = nIndex+n(i)-1;
end
end