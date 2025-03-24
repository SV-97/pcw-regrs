//! Python API for the `pcw_regrs` Rust crate. Please see the corresponding
//! documentation for more detailed information on the Rust internals.

use std::num::NonZeroUsize;
#[cfg(feature = "show_times")]
use std::time::Instant;

use pcw_regrs as rs;

use pcw_fn::{Functor, FunctorRef};

use numpy::PyReadonlyArray1;
use ordered_float::OrderedFloat;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyBytes};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod wrapper_types;
use wrapper_types::*;

type Float = f64;
type OFloat = OrderedFloat<f64>;

#[pyfunction]
// #[args(max_total_dof = "None", max_seg_dof = "None", weights = "None")]
pub fn fit_pcw_poly(
    sample_times: PyReadonlyArray1<Float>,
    response_values: PyReadonlyArray1<Float>,
    max_total_dof: Option<usize>,
    max_seg_dof: Option<usize>,
    weights: Option<PyReadonlyArray1<Float>>,
) -> Solution {
    rs::try_fit_pcw_poly(
        &rs::TimeSeriesSample::try_new(
            sample_times.as_slice().unwrap(),
            response_values.as_slice().unwrap(),
            weights.as_ref().map(|w| w.as_slice().unwrap()),
        )
        .unwrap(),
        &rs::UserParams {
            max_total_dof: max_total_dof.map(|u| {
                NonZeroUsize::new(u)
                    .expect("Invalid argument: total degrees of freedom have to be nonzero")
            }),
            max_seg_dof: max_seg_dof.map(|u| {
                NonZeroUsize::new(u)
                    .expect("Invalid argument: segment degrees of freedom have to be nonzero")
            }),
        },
    )
    .map(Solution::from_rs)
    .unwrap()
}

#[pyclass]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Solution {
    sol: Option<rs::Solution<OFloat>>,
}

impl Solution {
    pub fn from_rs(sol: rs::Solution<OFloat>) -> Self {
        Self { sol: Some(sol) }
    }
}

impl Solution {
    fn sol(&self) -> Option<rs::Solution<OFloat>> {
        self.sol.clone()
    }
}

#[pymethods]
impl Solution {
    #[new]
    pub fn new() -> Self {
        Self { sol: None }
    }

    #[cfg(feature = "serde")]
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Used in pickle/pickling
        let s = serde_json::to_string(&self).unwrap();
        Ok(PyBytes::new(py, s.as_bytes()))
    }

    #[cfg(feature = "serde")]
    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<&[u8]>(py) {
            Ok(s) => {
                *self = serde_json::from_slice(s).unwrap();

                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Return the best model w.r.t. the "one standard error" rule.
    pub fn ose_best(&self) -> PyResult<ScoredPolyModel> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => {
                let ose = sol.ose_best().unwrap();
                let scored_model = ScoredPolyModel::from_rs(ose);
                Ok(scored_model)
            }
        }
    }

    /// Return the best model w.r.t. the "x-times standard error" rule.
    pub fn xse_best(&self, x: Float) -> PyResult<ScoredPolyModel> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => {
                let xse = sol.xse_best(OrderedFloat(x)).unwrap();
                let scored_model = ScoredPolyModel::from_rs(xse);
                Ok(scored_model)
            }
        }
    }

    /// Return the global minimizer of the CV score.
    pub fn cv_minimizer(&self) -> PyResult<ScoredPolyModel> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => Ok(ScoredPolyModel::from_rs(sol.cv_minimizer().unwrap())),
        }
    }

    /// Return the models corresponding to the `n_best` lowest CV scores.
    pub fn n_cv_minimizers(&self, n_best: usize) -> PyResult<Vec<ScoredPolyModel>> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => Ok(sol
                .n_cv_minimizers(n_best)
                .unwrap()
                .fmap(ScoredPolyModel::from_rs)),
        }
    }

    /// Returns the optimal model corresponding to the given penalty γ
    pub fn model_for_penalty(&self, penalty: Float) -> PyResult<ScoredPolyModel> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => Ok(ScoredPolyModel::from_rs(
                sol.model_for_penalty(OrderedFloat(penalty)),
            )),
        }
    }

    /// The cross validation function mapping hyperparameters γ to CV scores.
    pub fn cv_func(&self) -> PyResult<PcwConstFn> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => Ok(PcwConstFn::from_rs(
                sol.cv_func().fmap_ref(|cv_se| cv_se.data),
            )),
        }
    }

    /// The cross validation function mapping hyperparameters γ to the standard errors
    /// of the CV scores.
    pub fn cv_se_func(&self) -> PyResult<PcwConstFn> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => Ok(PcwConstFn::from_rs(
                sol.cv_func().fmap_ref(|cv_se| cv_se.metadata),
            )),
        }
    }

    /// The cross validation function downsampled to the jumps of the model function.
    pub fn downsampled_cv_func(&self) -> PyResult<PcwConstFn> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => Ok(PcwConstFn::from_rs(
                sol.downsampled_cv_func().fmap_ref(|cv_se| cv_se.data),
            )),
        }
    }

    /// The cross validation standard error function downsampled to the jumps of the model function.
    pub fn downsampled_cv_se_func(&self) -> PyResult<PcwConstFn> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => Ok(PcwConstFn::from_rs(
                sol.downsampled_cv_func().fmap_ref(|cv_se| cv_se.metadata),
            )),
        }
    }

    /// The model function mapping hyperparameters γ to the corresponding solutions of
    /// the penalized partition problem.
    pub fn model_func(&self) -> PyResult<ModelFunc> {
        match self.sol() {
            None => Err(PyRuntimeError::new_err("Internal error.")),
            Some(sol) => Ok(ModelFunc::from_rs(sol.scored_model_func())), // PcwConstFn::from_rs( sol.downsampled_cv_func().fmap_ref(|cv_se| cv_se.metadata
        }
    }
}

/// Optimal (w.r.t. a cross-validation scheme) piecewise polynomial interpolation
#[pymodule]
fn pcw_regrs_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_pcw_poly, m)?)?;
    m.add_class::<Solution>()?;
    m.add_class::<ScoredPolyModel>()?;
    m.add_class::<PolyModelSpec>()?;
    m.add_class::<PcwConstFn>()?;
    Ok(())
}
