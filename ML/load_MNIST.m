

function [train_images,test_images,train_labels,test_labels] = load_MNIST()

train_images = loadMNISTImages('data\train-images.idx3-ubyte');
test_images = loadMNISTImages('data\t10k-images.idx3-ubyte');
train_labels = loadMNISTLabels('data\train-labels.idx1-ubyte');
test_labels = loadMNISTLabels('data\t10k-labels.idx1-ubyte');
