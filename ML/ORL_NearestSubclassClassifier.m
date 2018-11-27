clear

load('orl_train_test_data');
load('orl_train_test_lbls');

nClasses = 40;
nSubClasses = 2;
nTestImages = size(testData,2);
nTrainImages = size(trainData,2);
M = zeros(size(trainData,1), nClasses);

%kmeans for nSubClasses
% column(i:i+1) belongs to class K
centroids = zeros(size(trainData,1), nClasses*nSubClasses);
for j = 1:7:size(trainData,2)
    [idx,C] = kmeans(trainData(:,[j:j+6])', nSubClasses);
    i = ceil(j/7)+ceil(j/7)-1;
    centroids(:,i:i+1) = C';
end

%nearest centroid
dist = zeros(nTestImages, nClasses*nSubClasses);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses*nSubClasses
        dist(i,k) = norm(testData(:,i)-centroids(:,k),2)^2;
    end
    [~,resLabels(i)] = min(dist(i,:));
end

%convert reslabels to one class dimension.
for i = 1:length(resLabels)
   resLabels(i) = ceil(resLabels(i)/2)
end

%accuracy in %
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result and test labels
hold on
scatter(1:length(resLabels),resLabels, [], 'red')
scatter(1:length(testLbls),testLbls, [], 'blue')

