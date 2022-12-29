% Want to train the network so that it can realize the pictures of the same
% person but taken at a different time

%Problem: train image with larger or smaller size

clear all;clc;

% Import pic
net=squeezenet; %convolutional neural network
im=imread('nnproject.jpeg');% read the pixel of the picture
figure(1)
imshow(im);


% Resize

% Standard size
A=imread("face.jpg");
size_standard=size(A);

% Resize image for image that smaller than the standard size of the layer
for i=1:2
    if im(i)<size_standard(i) || im(i)>size_standard(i)    % Compare each number in the size vector
    
        im=imresize(im,[227 227]);
        break
    end
end

img_size=size(im);
img_size=img_size(1:2); %put the image size into an input vector 
analyzeNetwork(net); %displays a visualization of the network


%% Show activation of first layer

act1=activations(net,im,'conv1'); %return a 3-D array, 3rd dimension indexing the chanel for conv1
                                  %pass the image through channel 1
sz=size(act1);
act1=reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I=imtile(mat2gray(act1),'GridSize',[8 8]); %return to 4-D array , the third dimension represent image color
figure(2)
imshow(I)

%% Investigate the activation in specific channel
atch1ch20=act1(:,:,:,20); %choose channel 20
atch1ch20=mat2gray(atch1ch20); %Convert matrix to intensity image
atch1ch20=imresize(atch1ch20,img_size);
I=imtile({im,atch1ch20}); %reshape in channel 20 and show the activation
figure(3)
imshow(I)

%% Find the strongest activation channel

[maxValue,maxValueIndex]=max(max(max((act1))));
act1chMax=act1(:,:,:,maxValueIndex);
atch1chMax=mat2gray(act1chMax);
act1chMax=imresize(act1chMax,img_size);

I=imtile({im,atch1chMax});
figure(4)
imshow(I)

%% Investigate a deeper layer
act6=activations(net,im,'fire6-squeeze1x1');
sz=size(act6);
I=imtile(imresize(mat2gray(act6),[64 64]),'GridSize',[6 8]);
figure(5)
imshow(I)


act6relu=activations(net,im,'fire6-relu_squeeze1x1');
act6relu= reshape(act6relu,[sz(1) sz(2) 1 sz(3)]);

I=imtile(imresize(mat2gray(act6relu(:,:,:,[14 47])),img_size));%combine multiple into one rectangular im


figure(6)
imshow(I)

%% If the net realize the same person?

imtest=imread('imtest.png');
figure(7)
imshow(imtest)

% Resize image for image that smaller than the standard size of the layer
for i=1:3
    if imtest(i)<size_standard(i)     % Compare each number in the size vector
    
        imtest=imresize(imtest,[227 227]);
        break
    end
end

imgtest_size=size(imtest);
imgtest_size=imgtest_size(1:2);

act6test=activations(net,imtest,'fire6-relu_squeeze1x1');
sz=size(act6test);
act6test=reshape(act6test,[sz(1),sz(2),1,sz(3)]);

% 
% [maxValue,maxValueIndex]=max(max(max((act6test))));
% act6chMax=act6test(:,:,:,maxValueIndex);
% atch6chMax=mat2gray(act6chMax);
% act6chMax=imresize(act6chMax,imgtest_size);
% 
% I=imtile({imtest,atch6chMax});
% figure(8)
% imshow(I)

% Ploting
channeltest= repmat(imresize(mat2gray(act6test(:,:,:,[14 47])),imgtest_size),[1 1 3]);
channelyoung= repmat(imresize(mat2gray(act6relu(:,:,:,[14 47])),img_size),[1 1 3]);
I=imtile(cat(4,im,channelyoung*255,imtest,channeltest*255));
figure(9)
title('Imput Image, Channel 14, Channel 47')
imshow(I)






















