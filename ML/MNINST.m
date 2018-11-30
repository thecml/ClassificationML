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

%% Nearest Centroid Test
mu = train_nc(train_images, train_labels, nClasses, offset);
dist = zeros(nTestImages, nClasses);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses
        dist(i,k) = norm(test_images(:,i)-mu(:,k),2)^2;
    end 
    [~,resLabels(i)] = min(dist(i,:));
end
%subtract 1 to match test labels
resLabels = resLabels-1;

%accuracy in %
disp('MNIST NC accuracy:')
accuracy = sum(resLabels==test_labels)/nTestImages

%plot result labels
figure
scatter(1:length(resLabels),resLabels, [])
title('Plot of NC on the MNIST for 10 classes')
xlabel('N result label') 
ylabel('result label in class') 

%% Nearest Subclass Centroid Test - 2 subclasses
nSubClasses = 2;
centroids = train_nsc(train_images, train_labels, nClasses, nSubClasses);
dist = zeros(nTestImages, nClasses*nSubClasses);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses*nSubClasses
        dist(i,k) = norm(test_images(:,i)-centroids(:,k),2)^2;
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
disp('MNIST NSC-2 accuracy:')
accuracy = sum(resLabels==test_labels)/nTestImages

%plot result and test labels
figure
hold on
scatter(1:length(resLabels),resLabels, 50, 'red')
scatter(1:length(test_labels),test_labels, 5, 'blue')
title('Plot of NSC on MNIST for 10/2 classes')
xlabel('N result label') 
ylabel('result label in class') 

%% Nearest Subclass Centroid Test - 3 subclasses
nSubClasses = 3;
centroids = train_nsc(train_images, train_labels, nClasses, nSubClasses);
dist = zeros(nTestImages, nClasses*nSubClasses);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses*nSubClasses
        dist(i,k) = norm(test_images(:,i)-centroids(:,k),2)^2;
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
disp('MNIST NSC-3 accuracy:')
accuracy = sum(resLabels==test_labels)/nTestImages

%% Nearest Subclass Centroid Test - 5 subclasses
nSubClasses = 5;
centroids = train_nsc(train_images, train_labels, nClasses, nSubClasses);
dist = zeros(nTestImages, nClasses*nSubClasses);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses*nSubClasses
        dist(i,k) = norm(test_images(:,i)-centroids(:,k),2)^2;
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
disp('MNIST NSC-5 accuracy:')
accuracy = sum(resLabels==test_labels)/nTestImages

%% Nearest Neighbor Test
resLabels = train_nn(train_images, train_labels, test_images);

%accuracy in %
disp('MNIST NN accuracy:')
accuracy = sum(resLabels==test_labels)/nTestImages

%plot result and test labels
figure
hold on
scatter(1:length(resLabels),resLabels, 50, 'red')
scatter(1:length(test_labels),test_labels, 5, 'blue')
title('Plot of NN on MNIST for 10 classes')
xlabel('N result label') 
ylabel('result label in class') 

%% Perceptron Test BP
w = train_perceptron_backprop(train_images, train_labels, 0.01, nClasses);
test_tilde = [ones(1,size(test_images,2));test_images];
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
disp('MNIST PCEP-BP accuracy:')
accuracy = sum(resLabels==testLbls)/nTestImages

%plot result labels
figure
hold on
scatter(1:length(resLabels),resLabels)
title('Plot of Perceptron with BP on MNIST for 10 classes')
xlabel('N result label') 
ylabel('result label in class') 

%% Perceptron Test MSE
w = train_perceptron_mse(train_images, train_labels, nClasses, offset);
test_tilde = [ones(1,size(test_images,2));test_images];
resLabels = zeros(1, nTestImages);
for i = 1:nTestImages
    resClass = [];
    for k = 1:nClasses
        %calculate cost funciton and save it
        y = w(:,k)'*test_tilde(:,i);
        resClass = [resClass y]; 
    end
    %take the result which maximizes the cost function
    [~,resLabels(i)] = max(resClass);
end

%subtract 1 to match test labels
resLabels = resLabels'-1;

disp('MNIST PCEP-MSE accuracy:')    
accuracy = sum(resLabels==test_labels)/nTestImages

%plot result labels
figure
hold on
scatter(1:length(resLabels),resLabels)
title('Plot of Perceptron with MSE on MNIST for 10 classes')
xlabel('N result label') 
ylabel('result label in class') 




