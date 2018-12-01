clear

train_images = loadMNISTImages('data\train-images.idx3-ubyte');
test_images = loadMNISTImages('data\t10k-images.idx3-ubyte');
train_labels = loadMNISTLabels('data\train-labels.idx1-ubyte');
test_labels = loadMNISTLabels('data\t10k-labels.idx1-ubyte');

%% Global information for MNIST
nClasses = 10; % numbers 0-9
offset = 0;
nTrainImages = size(train_images,2);
nTestImages = size(test_images,2);
nPixels = size(train_images,1);

%% preprocessor - sort samples and labels in ascending order.
train_images = sortrows([train_images; train_labels']',nPixels+1);
test_images = sortrows([test_images; test_labels']',nPixels+1);
train_labels = sortrows(train_labels);
test_labels = sortrows(test_labels);
train_images = train_images(:,1:nPixels)';
test_images = test_images(:,1:nPixels)';

%% PCA for train data
pc_train = pca_reduce(train_images, 2);
figure
scatter(pc_train(1,:), pc_train(2,:), [], train_labels);
title('Scatter plot of PCA on the MNIST train set with D=2')
xlabel('PC1') 
ylabel('PC2')

%% PCA for test data
pc_test = pca_reduce(test_images, 2);
figure
scatter(pc_test(1,:), pc_test(2,:), [], test_labels);
title('Scatter plot of PCA on the MNIST test set with D=2')
xlabel('PC1') 
ylabel('PC2')

%% Nearest Centroid Test (PCA)
mu = train_nc(pc_train, train_labels, nClasses, offset);
dist = zeros(nTestImages, nClasses);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses
        dist(i,k) = norm(pc_test(:,i)-mu(:,k),2)^2;
    end 
    [~,resLabels(i)] = min(dist(i,:));
end
%subtract 1 to match test labels
resLabels = resLabels-1;

%accuracy in %
disp("MNIST NC PCA accuracy:")
accuracy = sum(resLabels==test_labels)/nTestImages

%% Nearest Subclass Centroid Test - 2 subclasses (PCA)
nSubClasses = 2;
centroids = train_nsc(pc_train, train_labels, nClasses, nSubClasses);
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

%subtract 1 to match test labels
resLabels = resLabels-1;

%accuracy in % for 10/2
disp("MNIST NSC-2 PCA accuracy:")
accuracy = sum(resLabels==test_labels)/nTestImages

%% Nearest Subclass Centroid Test - 3 subclasses (PCA)
nSubClasses = 3;
centroids = train_nsc(pc_train, train_labels, nClasses, nSubClasses);
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

%subtract 1 to match test labels
resLabels = resLabels-1;

%accuracy in % for 10/3
disp("MNIST NSC-3 PCA accuracy:")
accuracy = sum(resLabels==test_labels)/nTestImages

%% Nearest Subclass Centroid Test - 5 subclasses (PCA)
nSubClasses = 5;
centroids = train_nsc(pc_train, train_labels, nClasses, nSubClasses);
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

%subtract 1 to match test labels
resLabels = resLabels-1;

%accuracy in % for 10/5
disp("MNIST NSC-5 PCA accuracy:")
accuracy = sum(resLabels==test_labels)/nTestImages

%% Nearest Neighbor Test (PCA)
resLabels = train_nn(pc_train, train_labels, pc_test);

%accuracy in %
disp("MNIST NN PCA accuracy:")
accuracy = sum(resLabels==test_labels)/nTestImages

%% Perceptron Test BP (PCA)
w = train_perceptron_backprop(pc_train, train_labels, 0.1, nClasses);
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
disp("MNIST PCEP-BP PCA accuracy:")
accuracy = sum(resLabels==test_labels)/nTestImages

%% Perceptron Test MSE (PCA)
w = train_perceptron_mse(pc_train, train_labels, nClasses, offset);
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
disp("MNIST PCEP-MSE PCA accuracy:")
accuracy = sum(resLabels==test_labels)/nTestImages


