height = 480;
width = 640;
% itemsize = 4;

imgpath = 'F:/data/blender/monkey/data/%04d.jpg';
img = rgb2gray(imread(sprintf(imgpath, 150)));

imgGPU = GPUImage(height, width, 1, 1);
imgGPU.upload(img);

display('creating image model');
imodel = ImageModel(imgGPU);
imodel.configure();
imodel.compute();
display('elapsed time:')
imodel.elapsedTime()
display('image model created');

imgConstantGPU = GPUImage(10, 20, 1, 4);
imgGradientGPU = GPUImage(height, width, 2, 4);

imodel.getImageConstant(imgConstantGPU);
imodel.getImageGradient(imgGradientGPU);

imgConstant = imgConstantGPU.download();


figure; imshow(img, [0, 255]); colorbar(); colormap('jet'); title('input image');
figure; imshow(imgConstant, [0, 1]); colorbar(); colormap('jet'); title('img constant');
