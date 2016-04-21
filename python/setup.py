#!/usr/bin/env python

'''
    setup.py
    ----------

    Distutils setup script to build Python wrappers for the flofilter_gpu
    library.

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
'''


from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy as np

incDirs = ['/usr/local/include',
           '/usr/local/cuda/include',
           np.get_include()]

libDirs = ['/usr/local/lib']


libs = ['flowfilter_gpu']
cflags = ['-std=c++11']

cython_directives = {'embedsignature' : True}

def createExtension(name, sources):

    global incDirs
    global libDirs
    global libs

    ext = Extension(name,
                    sources=sources,
                    include_dirs=incDirs,
                    library_dirs=libDirs,
                    libraries=libs,
                    runtime_library_dirs=libs,
                    language='c++',
                    extra_compile_args=cflags)

    return ext

# list of extension modules
extensions = list()

#################################################
# PURE PYTHON PACKAGES
#################################################
py_packages = ['flowfilter', 'flowfilter.gpu', 'flowfilter.rsc']

# package data include Cython .pxd files
package_data = {'flowfilter' : ['*.pxd'],
                'flowfilter.gpu': ['*.pxd'],
                'flowfilter.rsc' : ['colorWheel.png']}

#################################################
# CYTHON EXTENSIONS
#################################################
GPUmodulesTable = [ ('flowfilter.image', ['flowfilter/image.pyx']),
                    ('flowfilter.gpu.image', ['flowfilter/gpu/image.pyx']),
                    ('flowfilter.gpu.imagemodel', ['flowfilter/gpu/imagemodel.pyx']),
                    ('flowfilter.gpu.pyramid', ['flowfilter/gpu/pyramid.pyx']),
                    ('flowfilter.gpu.propagation', ['flowfilter/gpu/propagation.pyx']),
                    ('flowfilter.gpu.update', ['flowfilter/gpu/update.pyx']),
                    ('flowfilter.gpu.display', ['flowfilter/gpu/display.pyx']),
                    ('flowfilter.gpu.camera', ['flowfilter/gpu/camera.pyx']),
                    ('flowfilter.gpu.rotation', ['flowfilter/gpu/rotation.pyx']),

                    # this module cannot be called flowfilter.gpu.flowfilter
                    ('flowfilter.gpu.flowfilters', ['flowfilter/gpu/flowfilters.pyx'])
                    ]

for mod in GPUmodulesTable:
    extList = cythonize(createExtension(mod[0], mod[1]), compiler_directives=cython_directives)
    extensions.extend(extList)


# call distutils setup
setup(name='flowfilter',
    version='0.1',
    author='Juan David Adarve',
    author_email='juanda0718@gmail.com',
    maintainer='Juan David Adarve',
    maintainer_email='juanda0718@gmail',
    url='https://github.com/jadarve/optical-flow-filter',
    description='A real time optical flow library.',
    license='3-clause BSD',
    packages=py_packages,
    ext_modules=extensions,
    package_data=package_data)
