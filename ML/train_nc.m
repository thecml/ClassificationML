function mu = train_nc(trainData, trainLbls, nClasses, offset) 

%Nearest Centroid
%n = num of samples in each training class
%find a matrix with the means of the classes using sorted data
mu = zeros(size(trainData,1), nClasses);
classes = 1:nClasses;
n = zeros(size(classes,1),size(classes,2))';
nIndex = 1;

for i = 1:nClasses
    n(i,:) = sum(trainLbls==classes(i));
    mu(:,i) = sum(trainData(:,nIndex:nIndex+n(i)-offset)')/n(i);
    nIndex = nIndex+n(i);
end
end