clear all;
close all;


% NOTE: change image path to corresponding 
imgpath = 'F:/data/blender/monkey/data/%04d.jpg';

f = FlowFilter(480, 640);
f.setMaxFlow(4);
f.setGamma(50);
f.setSmoothIterations(2);

% run the flow filter for 150 images, each time printing
% the elapsed time in milliseconds
for n = 0:150
	img = rgb2gray(imread(sprintf(imgpath, n)));
	% img = uint8(zeros(480, 640));
	f.loadImage(img);
	f.compute();
	f.elapsedTime()
end

% download image and show
imgd = f.downloadImage();
imshow(imgd);

% download flow and show each individual component
flow = f.downloadFlow();
fx = flow(:,:,1);
fy = flow(:,:,2);
figure; imshow(fx, [-3, 3]); colorbar(); colormap('jet'); title('Flow_X');
figure; imshow(fy, [-3, 3]); colorbar(); colormap('jet'); title('Flow_Y');
