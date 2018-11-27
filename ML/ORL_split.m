%load data

load('orl_data.mat');
load('orl_lbls.mat');

img1 = data(:,1);

%%imshow(reshape(img1,[40,30]));

Nlbls = 40;
dataSize = size(data,2)

NSamplesClass = dataSize/Nlbls;

pTrain = 0.7;
pTest  = 1-pTrain;

indecies = randperm(dataSize/Nlbls);

trainIndecies=[];
testIndecies=[];

for i = 0:Nlbls-1
    trainIndecies = [trainIndecies i*NSamplesClass+indecies(1:round(NSamplesClass*pTrain))]
    testIndecies = [testIndecies i*NSamplesClass+indecies(round(NSamplesClass*pTrain)+1:NSamplesClass)]
end

trainData = data(:,trainIndecies);
testData  = data(:,testIndecies);
trainLbls = lbls(trainIndecies);
testLbls = lbls(testIndecies);

save('orl_train_test_data.mat','trainData','testData');
save('orl_train_test_lbls.mat','trainLbls','testLbls');
