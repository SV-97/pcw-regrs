use super::annotate::Annotated;
use num_traits::real::Real;
use ordered_float::OrderedFloat;
use pcw_fn::VecPcwFn;
use std::{num::NonZeroUsize, ops::Index};
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub type Float = f64;
/// # Safety
/// This has to have the same layout as Float (transparent repr)
pub type OFloat = OrderedFloat<Float>;

// Maybe this should be a [std::num::NonZeroU64] instead.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DegreeOfFreedom(NonZeroUsize);

impl From<DegreeOfFreedom> for usize {
    fn from(value: DegreeOfFreedom) -> Self {
        value.0.into()
    }
}

impl From<DegreeOfFreedom> for NonZeroUsize {
    fn from(value: DegreeOfFreedom) -> Self {
        value.0
    }
}

impl TryFrom<usize> for DegreeOfFreedom {
    type Error = <NonZeroUsize as TryFrom<usize>>::Error;
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::try_from(value).map(DegreeOfFreedom)
    }
}

impl DegreeOfFreedom {
    pub fn one() -> Self {
        unsafe { DegreeOfFreedom(NonZeroUsize::new_unchecked(1)) }
    }

    /// Convert self to a polynomial degree
    pub fn to_deg(self) -> usize {
        usize::from(self.0) - 1
    }
}

/// A specification of a model on some segment of data given by the start and stop indices w.r.t. a timeseries.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SegmentModelSpec {
    /// The index of the point in time (in the original timeseries) at which this segment starts.
    pub start_idx: usize,
    /// The index of the point in time (in the original timeseries) at which this segment stops.
    pub stop_idx: usize,
    /// The model for the segment - so its dofs.
    pub seg_dof: DegreeOfFreedom,
}

/// Maps each penalty γ to the corresponding optimal models given as piecewise model
/// specifications: the arguments of the "inner" piecewise function are jump indices.
pub type ModelFunc = VecPcwFn<OFloat, VecPcwFn<usize, SegmentModelSpec>>;

/// A function mapping hyperparameter values to CV scores; each score being annotated with
/// its standard error.
pub type CvFunc = VecPcwFn<OFloat, Annotated<OFloat, OFloat>>;

#[derive(Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UserParams {
    pub max_total_dof: Option<NonZeroUsize>,
    pub max_seg_dof: Option<NonZeroUsize>,
}

impl UserParams {
    /// Returns None if input timeseries has length 0.
    pub fn match_to_timeseries(self, ts: &ValidTimeSeriesSample) -> MatchedUserParams {
        let data_len: NonZeroUsize = ts.len();
        let max_total_dof = if let Some(max_dof) = self.max_total_dof {
            std::cmp::min(data_len, max_dof)
        } else {
            data_len
        };
        let max_seg_dof = if let Some(max_dof) = self.max_seg_dof {
            std::cmp::min(max_total_dof, max_dof)
        } else {
            max_total_dof
        };
        MatchedUserParams {
            max_total_dof: DegreeOfFreedom(max_total_dof),
            max_seg_dof: DegreeOfFreedom(max_seg_dof),
        }
    }
}

pub struct MatchedUserParams {
    pub max_total_dof: DegreeOfFreedom,
    pub max_seg_dof: DegreeOfFreedom,
}

// TODO: add other lifetime params
pub struct TimeSeriesSample<'a> {
    /// Has to have same length as `response` and `weights`.
    times: &'a [Float],
    response: &'a [Float],
    weights: Option<&'a [Float]>,
}

impl<'a> TimeSeriesSample<'a> {
    pub fn try_new(
        times: &'a [Float],
        response: &'a [Float],
        weights: Option<&'a [Float]>,
    ) -> Option<Self> {
        if let Some(w) = weights && w.len() != response.len() {
            None
        } else if times.len() != response.len() {
            None
        } else {
            Some(TimeSeriesSample { times, response, weights })
        }
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

/// A nonempty, NaN-free time series sample
pub struct ValidTimeSeriesSample<'a> {
    /// Has to have same length as `response` and `weights` and be nonempty.
    times: &'a [OFloat],
    response: &'a [OFloat],
    weights: Option<&'a [OFloat]>,
}

impl<'a> ValidTimeSeriesSample<'a> {
    pub fn times(&self) -> &'a [OFloat] {
        self.times
    }
    pub fn response(&self) -> &'a [OFloat] {
        self.response
    }
    pub fn weights(&self) -> Option<&'a [OFloat]> {
        self.weights
    }
}

impl<'a> TryFrom<&'a TimeSeriesSample<'a>> for ValidTimeSeriesSample<'a> {
    type Error = FitError;

    fn try_from(timeseries_sample: &'a TimeSeriesSample<'a>) -> Result<Self, Self::Error> {
        if timeseries_sample.is_empty() {
            Err(FitError::EmptyData)
        } else if timeseries_sample.times.iter().any(|x| x.is_nan()) {
            Err(FitError::NanInTimes)
        } else if timeseries_sample.response.iter().any(|x| x.is_nan()) {
            Err(FitError::NanInResponses)
        } else if let Some(weights) = timeseries_sample.weights && weights.iter().any(|x| x.is_nan()) {
            Err(FitError::NanInWeights)
        } else {
            Ok(ValidTimeSeriesSample {
                times: unsafe { std::mem::transmute(timeseries_sample.times) },
                response: unsafe { std::mem::transmute(timeseries_sample.response) },
                weights: timeseries_sample.weights.map(|w| unsafe { std::mem::transmute(w) }),
            })
        }
    }
}

impl<'a> ValidTimeSeriesSample<'a> {
    pub fn len(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.times.len()) }
    }

    pub fn slice<Idx>(&self, index: Idx) -> Self
    where
        [OFloat]: Index<Idx, Output = [OFloat]>,
        Idx: Clone,
    {
        ValidTimeSeriesSample {
            times: &self.times[index.clone()],
            response: &self.response[index.clone()],
            weights: self.weights.map(|w| &w[index]),
        }
    }
}

/// The various kinds of errors that can happen during the polynomial fitting process.
#[derive(Error, Debug, PartialEq, Eq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FitError {
    #[error("encountered a floating point NaN in the sequence of sample times")]
    NanInTimes = 0b11,
    #[error("encountered a floating point NaN in the sequence of timeseries values / responses")]
    NanInResponses = 0b101,
    #[error("encountered a floating point NaN in the sequence of timeseries values / responses")]
    NanInWeights = 0b1001,
    #[error("can't fit a model to an empty timeseries")]
    EmptyData = 0b10000,
}

impl FitError {
    pub fn is_any_nan_err(self) -> bool {
        (self as u32) % 2 == 1
    }
}

/// Squared euclidean metric d(x,y)=‖x-y‖₂².
#[inline]
pub fn euclid_sq_metric<T: Real>(x: &T, y: &T) -> T {
    let d = *x - *y;
    // d * d
    d.powi(2)
}

#[macro_export]
macro_rules! dof {
    ($n: expr) => {
        $crate::prelude::DegreeOfFreedom::try_from($n).unwrap()
    };
}
