% GPUImage.m
% 
% Matlab wrapper for flowfilter::gpu::GPUImage class
%
% copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
% license 3-clause BSD, see LICENSE for more details
%

classdef GPUImage < handle

    properties (SetAccess = private, Hidden = true)
        objHandle; % Handle to the underlying C++ class instance
    end

    methods
        %% Constructor - Create a new C++ class instance
        function this = GPUImage(varargin)
            this.objHandle = GPUImage_mex('new', varargin{:});
        end

        %% Destructor - Destroy the C++ class instance
        function delete(this)
            GPUImage_mex('delete', this.objHandle);
        end

        function v = height(this)
            v = GPUImage_mex('height', this.objHandle);
        end

        function v = width(this)
            v = GPUImage_mex('width', this.objHandle);
        end

        function v = depth(this)
            v = GPUImage_mex('depth', this.objHandle);
        end

        function v = pitch(this)
            v = GPUImage_mex('pitch', this.objHandle);
        end

        function v = itemSize(this)
            v = GPUImage_mex('itemSize', this.objHandle);
        end

        function clear(this)
            GPUImage_mex('clear', this.objHandle);
        end

        function upload(this, img)

            % convert to row major order
            if ndims(img) == 3
                imgRowMajor = permute(img, [2 1 3]);
            else
                imgRowMajor = permute(img, [2 1]);
            end

            GPUImage_mex('upload', this.objHandle, imgRowMajor);
        end

        function img = download(this)
            img = GPUImage_mex('download', this.objHandle);

            % convert to column major order
            if ndims(img) == 3
                img = permute(img, [2 1 3]);
            else
                img = permute(img, [2 1]);
            end
        end

        function testTextureCreation(this)
            GPUImage_mex('testTextureCreation', this.objHandle)
        end

    end % methods
end % classdef
