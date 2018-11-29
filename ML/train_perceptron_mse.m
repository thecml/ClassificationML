function w = train_perceptron_mse(trainData, trainLbls, nClasses)

%preprocessor
%sort the train data samples and labels in ascending order.
nPixels = size(trainData,1);
trainData = sortrows([trainData; trainLbls']',nPixels+1);
trainLbls = sortrows(trainLbls);
trainData = trainData(:,1:nPixels)';

nTrainImages = size(trainData,2);
train_tilde = [ones(1,size(trainData,2));trainData];
w = rand(size(trainData,1)+1, nClasses);

%find pseudo-inverse
%X=pinv of train_tilde, b = [1,..1], a=weights
X = (train_tilde'*train_tilde)^-1*train_tilde';

%train perceptron
for k = 1:nClasses
    lbls = [];
    for (i = 1:nTrainImages)
        %create label vector for samples in class
        if(trainLbls(i) == k)
            lbls = [lbls 1];
        else
            lbls = [lbls -1];
        end
    end
    %find MSE soluion
    w(:,k) = X'*lbls';
end