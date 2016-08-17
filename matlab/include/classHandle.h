/*
 * This file was based on the work by Oliver Woodford for creating C++ wrappers for
 * Matlab mex functions.
 * http://au.mathworks.com/matlabcentral/fileexchange/38964-example-matlab-class-wrapper-for-a-c++-class
 */

#ifndef CLASSHANDLE_H_
#define CLASSHANDLE_H_

#include "mex.h"
#include <stdint.h>
#include <string>
#include <typeinfo>


template<class T>
class ClassHandle {

public:
    ClassHandle(T *ptr):
        __ptr(ptr),
        __name(typeid(T).name()) {
    }

    ~ClassHandle() {
        delete __ptr;
    }

    bool isValid() {
        return __name == typeid(T).name();
    }

    T* ptr() {
        return __ptr;
    }

    std::string typeName() {
        return __name;
    }

private:
    std::string __name;
    T* __ptr;
};


template<class T>
inline mxArray *convertPtr2Mat(T *ptr) {

    // creates a 1x1 uint64 matrix
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new ClassHandle<T>(ptr));
    return out;
}

template<class T>
inline ClassHandle<T> *convertMat2HandlePtr(const mxArray *in) {
    
    if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS || mxIsComplex(in)) {
        mexErrMsgTxt("Input must be a real uint64 scalar.");
    }

    ClassHandle<T> *ptr = reinterpret_cast<ClassHandle<T> *>(*((uint64_t *)mxGetData(in)));
    if (!ptr->isValid()) {
        mexPrintf("Handle not valid. Expecting \"%s\", got \"%s\"\n", typeid(T).name(), ptr->typeName().c_str());
        mexErrMsgTxt("Handle not valid");
    }

    return ptr;
}


template<class T>
inline T *convertMat2Ptr(const mxArray *in) {
    return convertMat2HandlePtr<T>(in)->ptr();
}


template<class T>
inline void destroyObject(const mxArray *in) {
   delete convertMat2HandlePtr<T>(in);
}


#endif /* CLASSHANDLE_H_ */
