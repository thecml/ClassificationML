clear all

load('data/orl_data.mat');
load('data/orl_lbls.mat');


img1 = reshape(data(:,121),[40,30]);
img2 = reshape(data(:,132),[40,30]);
img3 = reshape(data(:,143),[40,30]);
img4 = reshape(data(:,154),[40,30]);
img5 = reshape(data(:,165),[40,30]);
img6 = reshape(data(:,176),[40,30]);
img7 = reshape(data(:,187),[40,30]);
img8 = reshape(data(:,198),[40,30]);
img9 = reshape(data(:,209),[40,30]);
img10 = reshape(data(:,220),[40,30]);

image = [ img1 ones(40,2) img2 ones(40,2) img3 ones(40,2) img4 ones(40,2) img5 ; 
          ones(2,30*5+2*4);
          img6 ones(40,2) img7 ones(40,2) img8 ones(40,2) img9  ones(40,2) img10]

imshow(image);
%imwrite(image,'../Report/dataset/ORL.png')
