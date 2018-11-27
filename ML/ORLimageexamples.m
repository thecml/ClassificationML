clear all

load('../ORL/orl_data.mat');
load('../ORL/orl_lbls.mat');


img1 = reshape(data(:,1),[40,30]);
img2 = reshape(data(:,22),[40,30]);
img3 = reshape(data(:,33),[40,30]);
img4 = reshape(data(:,44),[40,30]);
img5 = reshape(data(:,55),[40,30]);
img6 = reshape(data(:,66),[40,30]);
img7 = reshape(data(:,77),[40,30]);
img8 = reshape(data(:,88),[40,30]);
img9 = reshape(data(:,99),[40,30]);
img10 = reshape(data(:,110),[40,30]);

image = [ img1 ones(40,2) img2 ones(40,2) img3 ones(40,2) img4 ones(40,2) img5 ; 
          ones(2,30*5+2*4);
          img6 ones(40,2) img7 ones(40,2) img8 ones(40,2) img9  ones(40,2) img10]

imshow(image);
imwrite(image,'../Report/dataset/ORL.png')
