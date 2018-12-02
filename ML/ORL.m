clear

load('orl_train_test_data');
load('orl_train_test_lbls');

%% Global information for ORL
nClasses = 40;
offset = 1;
nPixels = 1200;
nTrainImages = size(trainData,2);
nTestImages = size(testData,2);

%% preprocessor - sort samples and labels in ascending order.
trainData = sortrows([trainData; trainLbls']',nPixels+1);
testData = sortrows([testData; testLbls']',nPixels+1);
trainLbls = sortrows(trainLbls);
testLbls = sortrows(testLbls);
trainData = trainData(:,1:nPixels)';
testData = testData(:,1:nPixels)';

%% Nearest Centroid Test
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
disp("ORL NC PCA accuracy:")
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result labels
figure
scatter(1:length(resLabels),resLabels, [])
title('Plot of Nearest Cenroid on the ORL for 40 classes')
xlabel('N result label') 
ylabel('result label in class')

%% Nearest Subclass Centroid Test
nSubClasses = 2;
centroids = train_nsc(trainData, trainLbls, nClasses, nSubClasses);
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

%accuracy in % for 40
disp("ORL NSC PCA accuracy:")
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result and test labels
figure
hold on
scatter(1:length(resLabels),resLabels, 50, 'red')
scatter(1:length(testLbls),testLbls, 5, 'blue')
title('Plot of NSC on ORL for 40/2 classes')
xlabel('N result label') 
ylabel('result label in class') 

%% Nearest Neighbor Test
resLabels = train_nn(trainData, trainLbls, testData);

%accuracy in %
disp("ORL NN PCA accuracy:")
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result and test labels
figure
hold on
scatter(1:length(resLabels),resLabels, 50, 'red')
scatter(1:length(testLbls),testLbls, 5, 'blue')
title('Plot of NN on ORL for 40 classes')
xlabel('N result label') 
ylabel('result label in class') 

%% Perceptron Test BP
w = train_perceptron_backprop(trainData, trainLbls, 0.01, nClasses, offset);
resLabels = zeros(1, nTestImages);
test_tilde = [ones(1,size(testData,2));testData];
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

disp('ORL PCEP-BP accuracy:')    
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result labels
figure
hold on
scatter(1:length(resLabels),resLabels)
title('Plot of Perceptron with BP on ORL for 40 classes, ETA=0.1')
xlabel('N result label') 
ylabel('result label in class') 

%% Perceptron Test MSE
w = train_perceptron_mse(trainData, trainLbls, nClasses, offset);
test_tilde = [ones(1,size(testData,2));testData];
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

disp('ORL PCEP-MSE accuracy:')    
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result labels
figure
hold on
scatter(1:length(resLabels),resLabels)
title('Plot of Perceptron with MSE on ORL for 40 classes')
xlabel('N result label') 
ylabel('result label in class') 

