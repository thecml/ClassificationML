clear

load('orl_train_test_data');
load('orl_train_test_lbls');


pc = pca_reduce(trainData, 2);
scatter(pc(1,:), pc(2,:), [], trainLbls);






nClasses = 40;
nTrainImages = size(trainData,2);
nTestImages = size(testData,2);
eta = 0.01;








%NN TEST
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










%NC TEST
dist = zeros(nTestImages, nClasses);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses
        dist(i,k) = norm(test_images(:,i)-mu(:,k),2)^2;
    end 
    [~,resLabels(i)] = min(dist(i,:));
end

%accuracy in %
accuracy = sum(resLabels-1==test_labels)/nTestImages

%plot result labels
scatter(1:length(resLabels),resLabels, [])






%NSC TEST
%nearest centroid
centroids = train_nsc(trainData, trainLbls, 40, 2)
dist = zeros(nTestImages, nClasses*nSubClasses);
resLabels = zeros(nTestImages, 1);
for i = 1:nTestImages
    for k = 1:nClasses*nSubClasses
        dist(i,k) = norm(test_images(:,i)-centroids(:,k),2)^2;
    end
    [~,resLabels(i)] = min(dist(i,:));
end
%convert reslabels to one class dimension.
%also adjust labels to match 0:nClasses-1
for i = 1:length(resLabels)
   resLabels(i) = ceil(resLabels(i)/2)-1;
end
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













