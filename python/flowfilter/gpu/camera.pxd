"""
    flowfilter.gpu.camera
    ---------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cdef extern from 'flowfilter/gpu/camera.h' namespace 'flowfilter::gpu':
    
    ctypedef struct perspectiveCamera_cpp 'flowfilter::gpu::perspectiveCamera':
    
        float alphaX
        float alphaY
        float centerX
        float centerY


    perspectiveCamera_cpp createPerspectiveCamera_cpp 'flowfilter::gpu::createPerspectiveCamera'(
        const float focalLength, const int height, const int width,
        const float sensorHeight, const float sensorWidth)



cdef class PerspectiveCamera:
    cdef perspectiveCamera_cpp cam
