clear all

[trainData,testData,trainLbls,testLbls] = load_MNIST();


img1 = reshape(trainData(:,22),[28,28]);%
img2 = reshape(trainData(:,71),[28,28]);%
img3 = reshape(trainData(:,110),[28,28]);%
img4 = reshape(trainData(:,99),[28,28]);%
img5 = reshape(trainData(:,21),[28,28]);%
img6 = reshape(trainData(:,1),[28,28]);%
img7 = reshape(trainData(:,19),[28,28]);%
img8 = reshape(trainData(:,10000),[28,28]);
img9 = reshape(trainData(:,18),[28,28]);%
img10 = reshape(trainData(:,9000),[28,28]);%

image = [ img1 ones(28,2) img2 ones(28,2) img3 ones(28,2) img4 ones(28,2) img5 ; 
          ones(2,28*5+2*4);
          img6 ones(28,2) img7 ones(28,2) img8 ones(28,2) img9  ones(28,2) img10]

imshow(image);

imwrite(image,'../Report/dataset/MNIST.png')