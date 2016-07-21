% ImageModel.m
% 
% Matlab wrapper for flowfilter::gpu::ImageModel class
%
% copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
% license 3-clause BSD, see LICENSE for more details
% 

classdef ImageModel < handle

    properties (SetAccess = private, Hidden = true)
        objHandle; % Handle to the underlying C++ class instance
    end

    methods
        %% Constructor - Create a new C++ class instance
        function this = ImageModel(inputImage)
            this.objHandle = ImageModel_mex('new', inputImage.objHandle);
        end

        %% Destructor - Destroy the C++ class instance
        function delete(this)
            ImageModel_mex('delete', this.objHandle);
        end

        function configure(this)
            ImageModel_mex('configure', this.objHandle);
        end

        function compute(this)
            ImageModel_mex('compute', this.objHandle);
        end

        function et = elapsedTime(this)
            et = ImageModel_mex('elapsedTime', this.objHandle);
        end

        function setInputImage(this, inputImage)
            ImageModel_mex('setInputImage', this.objHandle, inputImage.objHandle)
        end

        function getImageConstant(this, imgConstant)
            ImageModel_mex('getImageConstant', this.objHandle, imgConstant.objHandle)
        end

        function getImageGradient(this, imgGradient)
            ImageModel_mex('getImageGradient', this.objHandle, imgGradient.objHandle)
        end

    end % methods
end % classdef
