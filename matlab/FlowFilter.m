% FlowFilter.m
% 
% Matlab wrapper for flowfilter::gpu::FlowFilter class
%
% copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
% license 3-clause BSD, see LICENSE for more details
% 

classdef FlowFilter < handle

    properties (SetAccess = private, Hidden = true)
        objHandle; % Handle to the underlying C++ class instance
        H;
        W;
    end

    methods
        %% Constructor - Create a new C++ class instance
        function this = FlowFilter(varargin)
            this.objHandle = FlowFilter_mex('new', varargin{:});

            % Keep height and with as Matlab state variables to
            % use in downloadFlow() to rearrange flow.
            this.H = FlowFilter_mex('height', this.objHandle);
            this.W = FlowFilter_mex('width', this.objHandle);
        end

        %% Destructor - Destroy the C++ class instance
        function delete(this)
            FlowFilter_mex('delete', this.objHandle);
        end

        function v = height(this)
            this.H = FlowFilter_mex('height', this.objHandle);
            v = this.H
        end

        function v = width(this)
            this.W = FlowFilter_mex('width', this.objHandle);
            v = this.W;
        end

        function configure(this)
            FlowFilter_mex('configure', this.objHandle);
        end

        function compute(this)
            FlowFilter_mex('compute', this.objHandle);
        end

        function et = elapsedTime(this)
            et = FlowFilter_mex('elapsedTime', this.objHandle);
        end

        function loadImage(this, img)

            % convert to row major order
            imgRowMajor = permute(img, [2 1]);
            FlowFilter_mex('loadImage', this.objHandle, imgRowMajor);
        end

        function img = downloadImage(this)

            img = FlowFilter_mex('downloadImage', this.objHandle);

            % convert to column major order
            img = permute(img, [2 1]);
        end

        function flow = downloadFlow(this)

            flow = FlowFilter_mex('downloadFlow', this.objHandle);

            % FIXME: There has to be a better way to do this!!!

            % reshape flow size to [depth, width, height]
            flow = reshape(flow, [2, this.W, this.H]);

            % extract each flow component and compute transpose (row to col major)
            flow_x = reshape(flow(1,:,:), this.W, this.H)';
            flow_y = reshape(flow(2,:,:), this.W, this.H)';

            % merge components
            flow = zeros(this.H, this.W, 2);
            flow(:,:,1) = flow_x;
            flow(:,:,2) = flow_y;
        end

        function setGamma(this, gamma)
            FlowFilter_mex('setGamma', this.objHandle, gamma);
        end

        function gamma = getGamma(this)
            gamma = FlowFilter_mex('getGamma', this.objHandle);
        end

        function setMaxFlow(this, maxFlow)
            FlowFilter_mex('setMaxFlow', this.objHandle, maxFlow);
        end

        function maxFlow = getMaxFlow(this)
            maxFlow = FlowFilter_mex('getMaxFlow', this.objHandle);
        end

        function setSmoothIterations(this, N)
            FlowFilter_mex('setSmoothIterations', this.objHandle, N);
        end

        function N = getSmoothIterations(this)
            N = FlowFilter_mex('getSmoothIterations', this.objHandle);
        end

        function setPropagationBorder(this, N)
            FlowFilter_mex('setPropagationBorder', this.objHandle, N);
        end

        function N = getPropagationBorder(this)
            N = FlowFilter_mex('getPropagationBorder', this.objHandle);
        end

        function N = getPropagationIterations(this)
            N = FlowFilter_mex('getPropagationIterations', this.objHandle);
        end


    end % methods
end % classdef
