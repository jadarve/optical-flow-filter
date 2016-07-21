
% A really ugly command!!! but it works :D
% mex -I'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/include' -I../include -Iinclude -L'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/lib/x64/' -lcuda -lcudart_static -lflowfilter_gpu GPUImage_mex.cpp src/imgutil.cpp
% mex -I'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/include' -I../include -Iinclude -L'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/lib/x64/' -lcuda -lcudart_static -lflowfilter_gpu FlowFilter_mex.cpp src/imgutil.cpp
% mex -I'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/include' -I../include -Iinclude -L'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/lib/x64/' -lcuda -lcudart_static -lflowfilter_gpu PyramidalFlowFilter_mex.cpp src/imgutil.cpp


% NOTE: Change the include and library paths to its corresponding
% 		values in your machine.

includes = {'-I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/include"' ...
			'-I"../include"' ...
			'-I"include"'};

libsPath = {'-L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/lib/x64/"'};

libs = {'-lcuda' ...
		'-lcudart_static' ...
		'-lflowfilter_gpu'};

srcs = {'src/imgutil.cpp'};


modules = {'GPUImage_mex.cpp' ...
		   'FlowFilter_mex.cpp' ...
		   'PyramidalFlowFilter_mex.cpp' ...
		   'ImageModel_mex.cpp'};


for n = 1 : length(modules)
	m = char(modules(n));
	disp(['BUILDING MODULE: ' m]);

	% ugly
	mex(char(includes(1)), char(includes(2)), char(includes(3)), ...
		char(libsPath(1)), char(libs(1)), char(libs(2)), char(libs(3)), m, char(srcs))

	fprintf('\n')
end
