% PyramidalFlowFilter.m
% 
% Matlab wrapper for flowfilter::gpu::PyramidalFlowFilter class
%
% copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
% license 3-clause BSD, see LICENSE for more details
%

classdef PyramidalFlowFilter < handle

    properties (SetAccess = private, Hidden = true)
        objHandle;  % Handle to the underlying C++ class instance.
        H;          % Image height.
        W;          % Image width.
    end

    methods
        %% Constructor - Create a new C++ class instance
        function this = PyramidalFlowFilter(varargin)
            this.objHandle = PyramidalFlowFilter_mex('new', varargin{:});

            % Keep height and with as Matlab state variables to
            % use in downloadFlow() to rearrange flow.
            this.H = PyramidalFlowFilter_mex('height', this.objHandle);
            this.W = PyramidalFlowFilter_mex('width', this.objHandle);
        end

        %% Destructor - Destroy the C++ class instance
        function delete(this)
            PyramidalFlowFilter_mex('delete', this.objHandle);
        end

        function v = height(this)
            this.H = PyramidalFlowFilter_mex('height', this.objHandle);
            v = this.H
        end

        function v = width(this)
            this.W = PyramidalFlowFilter_mex('width', this.objHandle);
            v = this.W;
        end

        function configure(this)
            PyramidalFlowFilter_mex('configure', this.objHandle);
        end

        function compute(this)
            PyramidalFlowFilter_mex('compute', this.objHandle);
        end

        function et = elapsedTime(this)
            et = PyramidalFlowFilter_mex('elapsedTime', this.objHandle);
        end

        function loadImage(this, img)

            % convert to row major order
            imgRowMajor = permute(img, [2 1]);
            PyramidalFlowFilter_mex('loadImage', this.objHandle, imgRowMajor);
        end

        function img = downloadImage(this)

            img = PyramidalFlowFilter_mex('downloadImage', this.objHandle);

            % convert to column major order
            img = permute(img, [2 1]);
        end

        function flow = downloadFlow(this)

            flow = PyramidalFlowFilter_mex('downloadFlow', this.objHandle);

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

        function setGamma(this, level, gamma)
            PyramidalFlowFilter_mex('setGamma', this.objHandle, level, gamma);
        end

        function gamma = getGamma(this, level)
            gamma = PyramidalFlowFilter_mex('getGamma', this.objHandle, level);
        end

        function setMaxFlow(this, maxFlow)
            PyramidalFlowFilter_mex('setMaxFlow', this.objHandle, maxFlow);
        end

        function maxFlow = getMaxFlow(this)
            maxFlow = PyramidalFlowFilter_mex('getMaxFlow', this.objHandle);
        end

        function setSmoothIterations(this, level, N)
            PyramidalFlowFilter_mex('setSmoothIterations', this.objHandle, level, N);
        end

        function N = getSmoothIterations(this, level)
            N = PyramidalFlowFilter_mex('getSmoothIterations', this.objHandle, level);
        end

        function setPropagationBorder(this, N)
            PyramidalFlowFilter_mex('setPropagationBorder', this.objHandle, N);
        end

        function N = getPropagationBorder(this)
            N = PyramidalFlowFilter_mex('getPropagationBorder', this.objHandle);
        end

        % function N = getPropagationIterations(this)
        %     N = PyramidalFlowFilter_mex('getPropagationIterations', this.objHandle);
        % end


    end % methods
end % classdef
