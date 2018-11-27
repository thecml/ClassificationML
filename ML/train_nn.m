function idx = train_nn(trainData, testData) 

nTrainImages = size(trainData,2);
nTestImages = size(testData,2);

dist = zeros(nTestImages, nTrainImages);
idx = [];
for i = 1:nTestImages
    for j = 1:nTrainImages
        %find the i closest to j
        dist(i,j) = norm(test_images(:,i)-train_images(:,j),2)^2;
    end
    [M,I] = min(dist(i,:));
    idx = [idx I];
end