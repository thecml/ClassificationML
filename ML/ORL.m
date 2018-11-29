clear

load('orl_train_test_data');
load('orl_train_test_lbls');

% Run PCA
%pc = pca_reduce(trainData, 2);
%figure
%scatter(pc(1,:), pc(2,:), [], trainLbls);
%title('Scatter plot of PCA on the ORL set with D=2')
%xlabel('PC1') 
%ylabel('PC2') 

% Global information for ORL
nClasses = 40;
offset = 1;
nPixels = 1200;
nTrainImages = size(trainData,2);
nTestImages = size(testData,2);

%preprocessor - sort samples and labels in ascending order.
trainData = sortrows([trainData; trainLbls']',nPixels+1);
testData = sortrows([testData; testLbls']',nPixels+1);
trainLbls = sortrows(trainLbls);
testLbls = sortrows(testLbls);
trainData = trainData(:,1:nPixels)';
testData = testData(:,1:nPixels)';

%NC TEST
dist = zeros(nTestImages, nClasses);
mu = train_nc(trainData, trainLbls, nClasses, offset);
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
figure
scatter(1:length(resLabels),resLabels, [])
title('Plot of Nearest Cenroid on the ORL for 40 classes')
xlabel('N result label') 
ylabel('result label in class')

% Nearest Subclass Centroid Test - 2 subclasses
nSubClasses = 2;
centroids = train_nsc(trainData, trainLbls, nClasses, nSubClasses)
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
   resLabels(i) = ceil(resLabels(i)/nSubClasses);
end

%accuracy in % for 40/2
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result and test labels
figure
hold on
scatter(1:length(resLabels),resLabels, 50, 'red')
scatter(1:length(testLbls),testLbls, 5, 'blue')
title('Plot of NSC on ORL for 40/2 classes')
xlabel('N result label') 
ylabel('result label in class') 









%NN TEST
idx = train_nn(trainData, testData);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for j = 1:nTrainImages
        %find the i closest to j
        dist(i,j) = norm(test_images(:,i)-train_images(:,j),2)^2;
    end
    [M,I] = min(dist(i,:));
    resLabels(i) = train_labels(I);
end

%accuracy in %
accuracy = sum(resLabels==test_labels)/nTestImages

%plot result and test labels
hold on
scatter(1:length(resLabels),resLabels, [], 'red')
scatter(1:length(test_labels),test_labels, [], 'blue')




















%perceptron TEST MSE
resLabels = zeros(1, nTestImages);
for i = 1:nTestImages
    resClass = [];
    for k = 1:nClasses
        %calculate decision funciton and save it
        y = w(:,k)'*test_tilde(:,i);
        resClass = [resClass y]; 
    end
    %take the result which maximizes the cost function
    [~,resLabels(i)] = max(resClass);
end
resLabels = resLabels';




%Percetron TEST BP
resLabels = zeros(1, nTestImages);
for i = 1:nTestImages
    resClass = [];
    for k = 1:nClasses
        %calculate decision funciton and save it
        y = w(:,k)'*test_tilde(:,i);
        resClass = [resClass y]; 
    end
    %take the result which maximizes the cost function
    [~,resLabels(i)] = max(resClass);
end

%accuracy in %
resLabels = resLabels';
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result labels
scatter(1:length(resLabels),resLabels)













