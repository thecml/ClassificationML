clear

train_images = loadMNISTImages('data\train-images.idx3-ubyte');
test_images = loadMNISTImages('data\t10k-images.idx3-ubyte');
train_labels = loadMNISTLabels('data\train-labels.idx1-ubyte');
test_labels = loadMNISTLabels('data\t10k-labels.idx1-ubyte');

%preprocessor
%sort the samples and labels in ascending order.
%train_images = sortrows([trainData; trainLbls']',nPixels+1);
%test_images = sortrows([trainData; trainLbls']',nPixels+1);
%train_labels = sortrows(trainLbls);
%test_labels = sortrows(test_labels);
%train_images = train_images(:,1:nPixels)';
%test_images = test_images(:,1:nPixels)';

pc = pca_reduce(train_images, 2);
scatter(pc(1,:), pc(2,:), [], train_labels);





%NSC TEST
%nearest centroid
centroids = train_nsc(train_images, train_labels, 10, 2)