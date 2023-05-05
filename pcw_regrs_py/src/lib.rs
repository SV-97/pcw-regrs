//! Python API for the `pcw_regrs` Rust crate. Please see the corresponding
//! documentation for more detailed information on the Rust internals.

#[cfg(feature = "show_times")]
use std::time::Instant;

use pcw_regrs as rs;

use pcw_fn::{Functor, FunctorRef, PcwFn};

use derive_new::new;
use numpy::{PyArray1, PyReadonlyArray1};
use ordered_float::OrderedFloat;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyBytes};

#[cfg(feature = "serde")]
use serde::{de, ser::SerializeStruct, Deserialize, Serialize};

type Float = f64;

/*
fn pyarray_to_solution(
    sample_times: PyReadonlyArray1<Float>,
    response_values: PyReadonlyArray1<Float>,
    max_total_dof: Option<usize>,
    max_seg_dof: Option<usize>,
) -> Result<rs::Solution<OrderedFloat<Float>>, rs::PolyFitError> {
    rs::fit_pcw_poly_primitive(
        sample_times.as_slice().unwrap(),
        response_values.as_slice().unwrap(),
        max_total_dof,
        max_seg_dof,
    )
}
*/

#[pyfunction]
// #[args(max_total_dof = "None", max_seg_dof = "None", weights = "None")]
pub fn fit_pcw_poly(
    sample_times: PyReadonlyArray1<Float>,
    response_values: PyReadonlyArray1<Float>,
    max_total_dof: Option<usize>,
    max_seg_dof: Option<usize>,
    weights: Option<PyReadonlyArray1<Float>>,
) -> Solution {
    rs::fit_pcw_poly_primitive(
        sample_times.as_slice().unwrap(),
        response_values.as_slice().unwrap(),
        max_total_dof,
        max_seg_dof,
        weights.as_ref().map(|w| w.as_slice().unwrap()),
    )
    .map(Solution::from_rs)
    .unwrap()
}

#[pyclass]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Solution {
    sol: Option<rs::SolutionCore<OrderedFloat<Float>, OrderedFloat<Float>>>,
}

impl Solution {
    pub fn from_rs(sol: rs::Solution<'static, OrderedFloat<Float>, OrderedFloat<Float>>) -> Self {
        Self {
            sol: Some(rs::SolutionCore::from(sol)),
        }
    }
}

impl Solution {
    fn sol<'a>(&'a self) -> Option<rs::Solution<'a, OrderedFloat<Float>, OrderedFloat<Float>>> {
        self.sol.as_ref().map(rs::Solution::from)
    }
}

#[pymethods]
impl Solution {
    #[new]
    pub fn new() -> Self {
        Self { sol: None }
    }

    #[cfg(feature = "serde")]
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        // Used in pickle/pickling
        let s = serde_json::to_string(&self).unwrap();
        Ok(PyBytes::new(py, s.as_bytes()))
    }

    #[cfg(feature = "serde")]
    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                *self = serde_json::from_slice(s.as_bytes()).unwrap();

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

    // /// The model function mapping hyperparameters γ to the corresponding solutions of
    // /// the penalized partition problem.
    // pub fn model_func(&self) -> &ModelFunc<E> {
    //     &self.model_func
    // }
}

#[pyclass]
// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct PolyModelSpec {
    #[pyo3(get)]
    start_idx: usize,
    #[pyo3(get)]
    stop_idx: usize,
    #[pyo3(get)]
    degrees_of_freedom: usize,
}

impl PolyModelSpec {
    // pub fn from_rs(sm: pcw_regrs::SegmentModelSpec<NonZeroUsize>) -> Self {
    //     PolyModelSpec {
    //         start_idx: sm.start_idx,
    //         stop_idx: sm.stop_idx,
    //         degrees_of_freedom: usize::from(sm.model),
    //     }
    // }
    pub fn from_rs<'a, R>(sm: pcw_regrs::SegmentModelSpec<rs::PolynomialArgs<'a, R>>) -> Self {
        PolyModelSpec {
            start_idx: sm.start_idx,
            stop_idx: sm.stop_idx,
            degrees_of_freedom: usize::from(sm.model.dof),
        }
    }
}

#[pyclass]
// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ScoredPolyModel {
    #[pyo3(get)]
    pub cv_score: Float,
    #[pyo3(get)]
    pub cut_idxs: Vec<usize>,
    #[pyo3(get)]
    pub model_params: Vec<PolyModelSpec>,
}

impl ScoredPolyModel {
    pub fn from_rs(scored_model: pcw_regrs::ScoredPolyModel<OrderedFloat<Float>>) -> Self {
        let pcw_regrs::ScoredPolyModel { model, score, .. } = scored_model;
        let (jumps, funcs) = model.into_jumps_and_funcs();
        ScoredPolyModel {
            cv_score: Float::from(score),
            cut_idxs: jumps.collect(),
            model_params: funcs.into_iter().map(PolyModelSpec::from_rs).collect(),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PcwConstFn {
    #[pyo3(get)]
    pub jump_points: Py<PyArray1<Float>>,
    #[pyo3(get)]
    pub values: Py<PyArray1<Float>>,
}

#[cfg(feature = "serde")]
impl Serialize for PcwConstFn {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("PcwConstFn", 2)?;
        Python::with_gil(|py| {
            let jump_points = unsafe { self.jump_points.as_ref(py).as_slice() }.unwrap();
            state.serialize_field("jump_points", jump_points)?;
            let values = unsafe { self.values.as_ref(py).as_slice() }.unwrap();
            state.serialize_field("values", values)?;
            Ok(())
        })?;
        state.end()
    }
}

#[cfg(feature = "serde")]
#[derive(new)]
struct PcwConstVisitor {}

#[cfg(feature = "serde")]
impl<'de> de::Visitor<'de> for PcwConstVisitor {
    type Value = PcwConstFn;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a PcwConstFn struct given by its `jump_points` and `values`.")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut jump_points: Option<Vec<Float>> = None;
        let mut values: Option<Vec<Float>> = None;
        while let Some(key) = map.next_key()? {
            match key {
                "jump_points" => {
                    if jump_points.is_some() {
                        return Err(de::Error::duplicate_field("jump_points"));
                    }
                    jump_points = Some(map.next_value()?);
                }
                "values" => {
                    if values.is_some() {
                        return Err(de::Error::duplicate_field("values"));
                    }
                    values = Some(map.next_value()?);
                }
                f => {
                    return Err(de::Error::unknown_field(f, &["jump_points", "values"]));
                }
            }
        }
        let jump_points = jump_points.ok_or_else(|| de::Error::missing_field("jump_points"))?;
        let values = values.ok_or_else(|| de::Error::missing_field("values"))?;
        Ok(PcwConstFn {
            jump_points: Python::with_gil(|py| PyArray1::from_vec(py, jump_points).into()),
            values: Python::with_gil(|py| PyArray1::from_vec(py, values).into()),
        })
    }
}

#[cfg(feature = "serde")]
impl<'de> de::Deserialize<'de> for PcwConstFn {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "PcwConstFn",
            &["jump_points", "values"],
            PcwConstVisitor::new(),
        )
    }
}

impl PcwConstFn {
    pub fn from_rs(pcw_fn: impl PcwFn<OrderedFloat<Float>, OrderedFloat<Float>>) -> Self {
        let (jumps, funcs) = pcw_fn.into_jumps_and_funcs();
        PcwConstFn {
            jump_points: Python::with_gil(|py| {
                PyArray1::from_vec(py, jumps.into_iter().map(Float::from).collect()).into()
            }),
            values: Python::with_gil(|py| {
                PyArray1::from_vec(py, funcs.into_iter().map(Float::from).collect()).into()
            }),
        }
    }
}

#[pymethods]
impl PcwConstFn {
    #[new]
    #[args(jump_points = "None", values = "None")]
    pub fn new(
        jump_points: Option<Py<PyArray1<Float>>>,
        values: Option<Py<PyArray1<Float>>>,
    ) -> PyResult<Self> {
        match (jump_points, values) {
            (None, None) => Ok(Self {
                jump_points: Python::with_gil(|py| unsafe { PyArray1::new(py, 0, false) }.into()),
                values: Python::with_gil(|py| unsafe { PyArray1::new(py, 0, false) }.into()),
            }),
            (Some(jump_points), Some(values)) => Ok(Self {
                jump_points,
                values,
            }), // TODO: add if jump_points.len() + 1 == values.len() guard
            _ => {
                panic!("Failed to construct `PcwConstFn`. Have to provide either both or no args.")
            }
        }
    }
}

/// Optimal (w.r.t. a cross-validation scheme) piecewise polynomial interpolation
#[pymodule]
fn pcw_regrs_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_pcw_poly, m)?)?;
    m.add_class::<Solution>()?;
    m.add_class::<ScoredPolyModel>()?;
    m.add_class::<PolyModelSpec>()?;
    m.add_class::<PcwConstFn>()?;
    Ok(())
}
