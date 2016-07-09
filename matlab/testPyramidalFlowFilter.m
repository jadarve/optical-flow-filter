clear all;
close all;

% NOTE: change image path to corresponding 
imgpath = 'F:/data/blender/monkey/data/%04d.jpg';

% Filter creation and parametrization (VGA with 2 pyramid levels)
f = PyramidalFlowFilter(480, 640, 2);
f.setMaxFlow(4);
f.setGamma(0, 50);
f.setGamma(1, 25);
f.setSmoothIterations(0, 2);
f.setSmoothIterations(1, 4);
f.setPropagationBorder(3);

% run the flow filter for 150 images, each time printing
% the elapsed time in milliseconds
for n = 0:150
	img = rgb2gray(imread(sprintf(imgpath, n)));
	f.loadImage(img);
	f.compute();
	disp(f.elapsedTime());
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
