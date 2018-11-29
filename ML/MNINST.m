clear

train_images = loadMNISTImages('data\train-images.idx3-ubyte');
test_images = loadMNISTImages('data\t10k-images.idx3-ubyte');
train_labels = loadMNISTLabels('data\train-labels.idx1-ubyte');
test_labels = loadMNISTLabels('data\t10k-labels.idx1-ubyte');

% Global information for MNIST
nClasses = 10; % numbers 0-9
offset = 0;
nTrainImages = size(train_images,2);
nTestImages = size(test_images,2);
nPixels = size(train_images,1);

% preprocessor - sort samples and labels in ascending order.
train_images = sortrows([train_images; train_labels']',nPixels+1);
test_images = sortrows([test_images; test_labels']',nPixels+1);
train_labels = sortrows(train_labels);
test_labels = sortrows(test_labels);
train_images = train_images(:,1:nPixels)';
test_images = test_images(:,1:nPixels)';

% PCA
%pc = pca_reduce(train_images, 2);
%figure
%scatter(pc(1,:), pc(2,:), [], train_labels);
%title('Scatter plot of PCA on the MNIST set with D=2')
%xlabel('PC1') 
%ylabel('PC2')

% Nearest Centroid Test
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
accuracy = sum(resLabels==test_labels)/nTestImages

%plot result labels
figure
scatter(1:length(resLabels),resLabels, [])
title('Plot of NC on the MNIST for 10 classes')
xlabel('N result label') 
ylabel('result label in class') 

% Nearest Subclass Centroid Test - 2 subclasses
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
accuracy = sum(resLabels==test_labels)/nTestImages

%plot result and test labels
figure
hold on
scatter(1:length(resLabels),resLabels, 50, 'red')
scatter(1:length(test_labels),test_labels, 5, 'blue')
title('Plot of NSC on MNIST for 10/2 classes')
xlabel('N result label') 
ylabel('result label in class') 

% Nearest Subclass Centroid Test - 3 subclasses
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
accuracy = sum(resLabels==test_labels)/nTestImages

% Nearest Subclass Centroid Test - 5 subclasses
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
accuracy = sum(resLabels==test_labels)/nTestImages














%NSC TEST
%nearest centroid
centroids = train_nsc(train_images, train_labels, 10, 2)