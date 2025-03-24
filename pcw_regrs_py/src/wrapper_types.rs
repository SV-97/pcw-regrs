use super::{rs, Float, OFloat};

use pcw_fn::PcwFn;

use derive_new::new;
use numpy::PyArray1;
use numpy::{ndarray::Array1, PyArrayMethods};
use pyo3::prelude::*;

#[cfg(feature = "serde")]
use serde::{de, ser::SerializeStruct, Serialize};

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

#[pymethods]
impl PolyModelSpec {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!(
            "PolyModelSpec(start_idx={}, stop_idx={}, degrees_of_freedom={})",
            self.start_idx, self.stop_idx, self.degrees_of_freedom
        )
    }
}

impl PolyModelSpec {
    // pub fn from_rs(sm: rs::SegmentModelSpec<NonZeroUsize>) -> Self {
    //     PolyModelSpec {
    //         start_idx: sm.start_idx,
    //         stop_idx: sm.stop_idx,
    //         degrees_of_freedom: usize::from(sm.model),
    //     }
    // }
    pub fn from_rs(sm: rs::SegmentModelSpec) -> Self {
        PolyModelSpec {
            start_idx: sm.start_idx,
            stop_idx: sm.stop_idx,
            degrees_of_freedom: usize::from(sm.seg_dof),
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

#[pymethods]
impl ScoredPolyModel {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!(
            "PolyModelSpec(cv_score={}, cut_idxs={}, model_params={})",
            self.cv_score,
            self.cut_idxs
                .iter()
                .fold("[".to_string(), |cur, elem| match cur.as_str() {
                    "[" => cur + &format!("{:#?}", elem),
                    _ => cur + &format!(", {:#?}", elem),
                })
                + "]",
            self.model_params
                .iter()
                .fold("[".to_string(), |cur, elem| match cur.as_str() {
                    "[" => cur + &elem.__repr__().to_string(),
                    _ => cur + &format!(", {}", elem.__repr__()),
                })
                + "]",
        )
    }
}

impl ScoredPolyModel {
    pub fn from_rs(scored_model: rs::ScoredModel<OFloat>) -> Self {
        let rs::ScoredModel { model, score, .. } = scored_model;
        let (jumps, funcs) = model.into_jumps_and_funcs();
        ScoredPolyModel {
            cv_score: Float::from(score),
            cut_idxs: jumps.collect(),
            model_params: funcs.into_iter().map(PolyModelSpec::from_rs).collect(),
        }
    }
}

#[pyclass]
#[derive(Debug)]
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
            let jump_points = unsafe { self.jump_points.bind(py).as_slice() }.unwrap();
            state.serialize_field("jump_points", jump_points)?;
            let values = unsafe { self.values.bind(py).as_slice() }.unwrap();
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
    pub fn from_rs(pcw_fn: impl PcwFn<OFloat, OFloat>) -> Self {
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
    #[pyo3(signature=(jump_points = None, values = None))]
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

#[pyclass]
#[derive(Debug)]
pub struct ModelFunc {
    #[pyo3(get)]
    /// the penalty parameters where one model changes into another one
    pub jump_points: Py<PyArray1<Float>>,
    #[pyo3(get)]
    /// the models corresponding to the penalties. The objects are instances of [ScoredPolyModel]
    pub values: Py<PyArray1<PyObject>>,
}

impl ModelFunc {
    pub fn from_rs(pcw_fn: impl PcwFn<OFloat, rs::ScoredModel<OFloat>>) -> Self {
        let (jumps, funcs) = pcw_fn.into_jumps_and_funcs();
        ModelFunc {
            jump_points: Python::with_gil(|py| {
                PyArray1::from_vec(py, jumps.into_iter().map(Float::from).collect()).into()
            }),
            values: Python::with_gil(|py| {
                PyArray1::from_owned_object_array(
                    py,
                    Array1::from_vec(
                        funcs
                            .into_iter()
                            .map(ScoredPolyModel::from_rs)
                            .map(|model| Py::new(py, model).unwrap())
                            .collect(),
                    ),
                )
                .into()
            }),
        }
    }
}
