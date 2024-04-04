use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

pub use torch_sys;
use torch_sys::python::{self, C_pyobject};

use crate::{TchError, Tensor};

pub type CPyObject = C_pyobject;

/// Check whether an object is a wrapped tensor or not.
///
/// # Safety
/// Undefined behavior if the given pointer is not a valid PyObject.
pub unsafe fn pyobject_check(pyobject: *mut CPyObject) -> Result<bool, TchError> {
    let v = unsafe_torch_err!(python::thp_variable_check(pyobject));
    Ok(v)
}

impl Tensor {
    /// Wrap a tensor in a Python object.
    pub fn pyobject_wrap(&self) -> Result<*mut CPyObject, TchError> {
        let v = unsafe_torch_err!(python::thp_variable_wrap(self.c_tensor));
        Ok(v)
    }

    /// Unwrap a tensor stored in a Python object. This returns `Ok(None)` if
    /// the object is not a wrapped tensor.
    ///
    /// # Safety
    /// Undefined behavior if the given pointer is not a valid PyObject.
    pub unsafe fn pyobject_unpack(pyobject: *mut CPyObject) -> Result<Option<Self>, TchError> {
        if !pyobject_check(pyobject)? {
            return Ok(None);
        }
        let v = unsafe_torch_err!(python::thp_variable_unpack(pyobject));
        Ok(Some(Tensor::from_ptr(v)))
    }
}

pub struct PyTensor(pub Tensor);

impl std::ops::Deref for PyTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn wrap_tch_err(err: TchError) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

impl<'source> FromPyObject<'source> for PyTensor {
    fn extract_bound(ob: &Bound<'source, PyAny>)  -> PyResult<Self> {
        let ptr = ob.as_ptr() as *mut CPyObject;
        let tensor = unsafe { Tensor::pyobject_unpack(ptr) };
        tensor
            .map_err(wrap_tch_err)?
            .ok_or_else(|| {
                let type_ = ob.get_type();
                PyErr::new::<PyTypeError, _>(format!("expected a torch.Tensor, got {type_}"))
            })
            .map(PyTensor)
    }
}

impl IntoPy<PyObject> for PyTensor {
    fn into_py(self, py: Python<'_>) -> PyObject {
        // There is no fallible alternative to ToPyObject/IntoPy at the moment, so we return
        // None on errors. https://github.com/PyO3/pyo3/issues/1813
        self.0.pyobject_wrap().map_or_else(
            |_| py.None(),
            |ptr| unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) },
        )
    }
}
