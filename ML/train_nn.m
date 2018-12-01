function resLabels = train_nn(trainData, trainLbls, testData) 

nTrainImages = size(trainData,2);
nTestImages = size(testData,2);
resLabels = zeros(nTestImages, 1);

dist = zeros(nTestImages, nTrainImages);
for i = 1:nTestImages
    for j = 1:nTrainImages
        %find the i closest to j
        dist(i,j) = norm(testData(:,i)-trainData(:,j),2)^2;
    end
    [M,I] = min(dist(i,:));
    resLabels(i) = trainLbls(I);
end