height = 4;
width = 10;
depth = 1;
itemsize = 4;

img = GPUImage(height, width, depth, itemsize);

% convert to float32
% A = single(zeros(480, 640, 1));
A = single(rand(height, width, depth));

img.upload(A);

B = img.download();

A
B

display('Test texture creation');

img.testTextureCreation();

display('completed...');
