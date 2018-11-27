clear

load('orl_train_test_data');
load('orl_train_test_lbls');

nClasses = 40;
nTestImages = size(testData,2);
mu = zeros(size(trainData,1), nClasses);

for j = 1:7:size(trainData,2)
    mu(:,ceil(j/7)) = mean(trainData(:,[j:j+6])')';
end

dist = zeros(nTestImages, nClasses);
resLabels = zeros(nTestImages, 1);

for i = 1:nTestImages
    for k = 1:nClasses
        dist(i,k) = norm(testData(:,i)-mu(:,k),2)^2;
    end 
    [~,resLabels(i)] = min(dist(i,:));
end

%accuracy in %
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result labels
scatter(1:length(resLabels),resLabels)
