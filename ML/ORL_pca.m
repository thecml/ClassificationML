clear

load('orl_train_test_data');
load('orl_train_test_lbls');

%%  Global information for ORL
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

%% PCA for train data
pc_train = pca_reduce(trainData, 2);
figure
scatter(pc_train(1,:), pc_train(2,:), [], trainLbls);
title('Scatter plot of PCA on the ORL train set with D=2')
xlabel('PC1') 
ylabel('PC2')

%% PCA for test data
pc_test = pca_reduce(testData, 2);
figure
scatter(pc_test(1,:), pc_test(2,:), [], testLbls);
title('Scatter plot of PCA on the ORL test set with D=2')
xlabel('PC1') 
ylabel('PC2')

%% Nearest Centroid Test (PCA)
dist = zeros(nTestImages, nClasses);
mu = train_nc(pc_train, trainLbls, nClasses, offset);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses
        dist(i,k) = norm(pc_test(:,i)-mu(:,k),2)^2;
    end 
    [~,resLabels(i)] = min(dist(i,:));
end

disp("ORL NC PCA accuracy:")
accuracy = sum(resLabels==testLbls)/nTestImages

%% Nearest Subclass Centroid Test (PCA)
nSubClasses = 2;
centroids = train_nsc(pc_train, trainLbls, nClasses, nSubClasses);
dist = zeros(nTestImages, nClasses*nSubClasses);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses*nSubClasses
        dist(i,k) = norm(pc_test(:,i)-centroids(:,k),2)^2;
    end
    [~,resLabels(i)] = min(dist(i,:));
end

%convert reslabels to one class dimension.
for i = 1:length(resLabels)
   resLabels(i) = ceil(resLabels(i)/nSubClasses);
end

disp("ORL NSC PCA accuracy:")
accuracy = sum(resLabels==testLbls)/nTestImages

%% Nearest Neighbor Test (PCA)
resLabels = train_nn(pc_train, trainLbls, pc_test);

%accuracy in %
disp("ORL NN PCA accuracy:")
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result and test labels
figure
hold on
scatter(1:length(resLabels),resLabels, 50, 'red')
scatter(1:length(testLbls),testLbls, 5, 'blue')
title('Plot of NN on ORL for 40 classes with PCA')
xlabel('N result label') 
ylabel('result label in class') 

%% Perceptron Test BP (PCA)
w = train_perceptron_backprop(pc_train, trainLbls, 1, nClasses, offset);
test_tilde = [ones(1,size(pc_test,2));pc_test];
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
disp('ORL PCEP-BP accuracy:')    
accuracy = sum(resLabels==testLbls)/nTestImages

%plot perceptron
figure
hold on
scatter(test_tilde(1,:),test_tilde(2,:),[],testLbls)
for i = 1:size(w,2)
    plotpc(w(2:end,i)',w(1,i));
end

%% Perceptron Test MSE (PCA)
w = train_perceptron_mse(pc_train, trainLbls, nClasses, offset);
test_tilde = [ones(1,size(pc_test,2));pc_test];
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
disp('ORL PCEP-MSE accuracy:')    
accuracy = sum(resLabels==testLbls)/nTestImages

%plot perceptron
figure
hold on
scatter(test_tilde(1,:),test_tilde(2,:),[],testLbls)
for i = 1:size(w,2)
    plotpc(w(2:end,i)',w(1,i));
end


