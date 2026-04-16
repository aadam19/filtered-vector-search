#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <limits>

namespace {

static PyObject* filter_topk(PyObject* /*self*/, PyObject* args) {
    PyObject* i_obj = nullptr;
    PyObject* d_obj = nullptr;
    PyObject* attr_obj = nullptr;
    int lo = 0;
    int hi = 0;
    int k = 0;

    if (!PyArg_ParseTuple(args, "OOOiii", &i_obj, &d_obj, &attr_obj, &lo, &hi, &k)) {
        return nullptr;
    }

    if (k < 0) {
        PyErr_SetString(PyExc_ValueError, "k must be non-negative");
        return nullptr;
    }

    PyArrayObject* I = (PyArrayObject*)PyArray_FROM_OTF(i_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* D = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* A = (PyArrayObject*)PyArray_FROM_OTF(attr_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!I || !D || !A) {
        Py_XDECREF(I);
        Py_XDECREF(D);
        Py_XDECREF(A);
        return nullptr;
    }

    if (PyArray_NDIM(I) != 1 || PyArray_NDIM(D) != 1 || PyArray_NDIM(A) != 1) {
        PyErr_SetString(PyExc_ValueError, "I, D, and attr must be 1D arrays");
        Py_DECREF(I);
        Py_DECREF(D);
        Py_DECREF(A);
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(I, 0);
    if (PyArray_DIM(D, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "I and D must have the same length");
        Py_DECREF(I);
        Py_DECREF(D);
        Py_DECREF(A);
        return nullptr;
    }

    npy_intp out_dim[1] = {k};
    PyArrayObject* D_out = (PyArrayObject*)PyArray_SimpleNew(1, out_dim, NPY_FLOAT32);
    PyArrayObject* I_out = (PyArrayObject*)PyArray_SimpleNew(1, out_dim, NPY_INT64);
    if (!D_out || !I_out) {
        Py_XDECREF(D_out);
        Py_XDECREF(I_out);
        Py_DECREF(I);
        Py_DECREF(D);
        Py_DECREF(A);
        return PyErr_NoMemory();
    }

    auto* i_ptr = (int64_t*)PyArray_DATA(I);
    auto* d_ptr = (float*)PyArray_DATA(D);
    auto* a_ptr = (int32_t*)PyArray_DATA(A);
    auto* dout = (float*)PyArray_DATA(D_out);
    auto* iout = (int64_t*)PyArray_DATA(I_out);

    const npy_intp attr_n = PyArray_DIM(A, 0);
    const float inf = std::numeric_limits<float>::infinity();
    for (int t = 0; t < k; ++t) {
        dout[t] = inf;
        iout[t] = -1;
    }

    int out_idx = 0;
    for (npy_intp j = 0; j < n && out_idx < k; ++j) {
        const int64_t id = i_ptr[j];
        if (id < 0 || id >= attr_n) {
            continue;
        }
        const int32_t v = a_ptr[id];
        if (v >= lo && v <= hi) {
            dout[out_idx] = d_ptr[j];
            iout[out_idx] = id;
            ++out_idx;
        }
    }

    Py_DECREF(I);
    Py_DECREF(D);
    Py_DECREF(A);
    return Py_BuildValue("NN", D_out, I_out);
}

static PyMethodDef Methods[] = {
    {"filter_topk", (PyCFunction)filter_topk, METH_VARARGS,
     "Filter candidate ids by attribute range and keep first k."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "filter_kernel",
    "C++ filter kernel helpers.",
    -1,
    Methods
};

} // namespace

PyMODINIT_FUNC PyInit_filter_kernel(void) {
    import_array();
    return PyModule_Create(&ModuleDef);
}
