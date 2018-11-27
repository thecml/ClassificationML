clear

load('orl_train_test_data');
load('orl_train_test_lbls');

nClasses = 40;
nTestImages = size(testData,2);
nTrainImages = size(trainData,2);

dist = zeros(nTestImages, nTrainImages);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for j = 1:nTrainImages
        %find the i closest to j
        dist(i,j) = norm(testData(:,i)-trainData(:,j),2)^2;
    end
    [M,I] = min(dist(i,:));
    resLabels(i) = trainLbls(I);
end

%accuracy
accuracy = sum(resLabels==testLbls)/nTestImages

hold on
scatter(1:length(resLabels),resLabels, [], 'red')
scatter(1:length(testLbls),testLbls, [], 'blue')
