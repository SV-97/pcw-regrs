use super::annotate::Annotated;
use indoc::indoc;
use num_traits::{real::Real, Float, FromPrimitive};
use ordered_float::OrderedFloat;
use pcw_fn::VecPcwFn;
use std::{num::NonZeroUsize, ops::Index};
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A marker trait that we use to wrap all the properties we internally need of a floating point type
pub trait RawFloat:
    Float + 'static + Send + Sync + FromPrimitive + std::iter::Sum + Default
{
}

/// # Safety
/// The `Base` type has to have the same layout as the type implementing this trait. That is the
/// assertion
/// `assert_eq!(std::alloc::Layout::<T>::new(), std::alloc::Layout::<<T as OrdFloat>::Base>::new());`
/// must hold whenever `T: OrdFloat`.
pub unsafe trait OrdFloat: RawFloat + Ord {
    // Float + 'static + Send + FromPrimitive
    type Base: RawFloat;
}

impl RawFloat for f64 {}
impl RawFloat for OrderedFloat<f64> {}
unsafe impl OrdFloat for OrderedFloat<f64> {
    /// # Safety
    /// OrderedFloat has transparent repr so we're guaranteed the same layout between the wrapped and unwrapped versions
    type Base = f64;
}

impl RawFloat for f32 {}
impl RawFloat for OrderedFloat<f32> {}
unsafe impl OrdFloat for OrderedFloat<f32> {
    /// # Safety
    /// OrderedFloat has transparent repr so we're guaranteed the same layout between the wrapped and unwrapped versions
    type Base = f32;
}

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
    #[inline(always)]
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::try_from(value).map(DegreeOfFreedom)
    }
}

impl DegreeOfFreedom {
    #[inline]
    pub const fn new(n: usize) -> Self {
        if n == 0 {
            panic!("Degrees of freedom can't be 0")
        } else {
            DegreeOfFreedom(unsafe { NonZeroUsize::new_unchecked(n) })
        }
    }

    /// # Safety
    /// The parameter [n] has to be nonzero
    #[inline(always)]
    pub unsafe fn new_unchecked(n: usize) -> Self {
        DegreeOfFreedom(unsafe { NonZeroUsize::new_unchecked(n) })
    }

    #[inline(always)]
    pub const fn one() -> Self {
        unsafe { DegreeOfFreedom(NonZeroUsize::new_unchecked(1)) }
    }

    /// Convert self to a polynomial degree
    #[inline(always)]
    pub fn to_deg(self) -> usize {
        usize::from(self.0) - 1
    }

    #[inline]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        usize::from(self)
            .checked_sub(rhs.into())
            .and_then(|x| Self::try_from(x).ok())
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
pub type ModelFunc<T> = VecPcwFn<T, VecPcwFn<usize, SegmentModelSpec>>;

/// A function mapping hyperparameter values to CV scores; each score being annotated with
/// its standard error.
pub type CvFunc<T> = VecPcwFn<T, Annotated<T, T>>;

#[derive(Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UserParams {
    pub max_total_dof: Option<NonZeroUsize>,
    pub max_seg_dof: Option<NonZeroUsize>,
}

impl UserParams {
    /// Returns None if input timeseries has length 0.
    pub fn match_to_timeseries<T>(self, ts: &ValidTimeSeriesSample<T>) -> MatchedUserParams
    where
        T: OrdFloat,
    {
        let data_len: NonZeroUsize = ts.len();
        let len_constraint = if cfg!(feature = "dofs-sub-one") {
            NonZeroUsize::new(usize::from(ts.len()) - 1)
                .expect(indoc! {"
                    Invalid combination of configuration and input: the input
                    timeseries has length 1 but the current configuration is such that at most
                    length - 1 degrees of freedom may be used. However there are no models with 0 degrees of freedom.
                "})
        } else {
            data_len
        };
        let max_total_dof = if let Some(max_dof) = self.max_total_dof {
            std::cmp::min(len_constraint, max_dof)
        } else {
            len_constraint
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

#[derive(Debug, Clone, PartialEq)]
// TODO: add other lifetime params
pub struct TimeSeriesSample<'a, T: Float> {
    /// Has to have same length as `response` and `weights`.
    times: &'a [T],
    response: &'a [T],
    weights: Option<&'a [T]>,
}

impl<'a, T> TimeSeriesSample<'a, T>
where
    T: Float,
{
    pub fn try_new(times: &'a [T], response: &'a [T], weights: Option<&'a [T]>) -> Option<Self> {
        if let Some(w) = weights
            && w.len() != response.len()
        {
            None
        } else if times.len() != response.len() {
            None
        } else {
            Some(TimeSeriesSample {
                times,
                response,
                weights,
            })
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
pub struct ValidTimeSeriesSample<'a, T>
where
    T: OrdFloat,
{
    /// Has to have same length as `response` and `weights` and be nonempty.
    times: &'a [T],
    response: &'a [T],
    weights: Option<&'a [T]>,
}

impl<'a, T> ValidTimeSeriesSample<'a, T>
where
    T: OrdFloat,
{
    pub fn times(&self) -> &'a [T] {
        self.times
    }
    pub fn response(&self) -> &'a [T] {
        self.response
    }
    pub fn weights(&self) -> Option<&'a [T]> {
        self.weights
    }
}

impl<'a, T> TryFrom<&'a TimeSeriesSample<'a, T::Base>> for ValidTimeSeriesSample<'a, T>
where
    T: OrdFloat,
{
    type Error = FitError;

    fn try_from(timeseries_sample: &'a TimeSeriesSample<'a, T::Base>) -> Result<Self, Self::Error> {
        if timeseries_sample.is_empty() {
            Err(FitError::EmptyData)
        } else if timeseries_sample.times.iter().any(|x| x.is_nan()) {
            Err(FitError::NanInTimes)
        } else if timeseries_sample.response.iter().any(|x| x.is_nan()) {
            Err(FitError::NanInResponses)
        } else if let Some(weights) = timeseries_sample.weights
            && weights.iter().any(|x| x.is_nan())
        {
            Err(FitError::NanInWeights)
        } else {
            Ok(ValidTimeSeriesSample {
                times: unsafe {
                    std::mem::transmute::<&'a [T::Base], &'a [T]>(timeseries_sample.times)
                },
                response: unsafe {
                    std::mem::transmute::<&'a [T::Base], &'a [T]>(timeseries_sample.response)
                },
                weights: timeseries_sample
                    .weights
                    .map(|w| unsafe { std::mem::transmute(w) }),
            })
        }
    }
}

impl<T> ValidTimeSeriesSample<'_, T>
where
    T: OrdFloat,
{
    pub fn len(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.times.len()) }
    }

    pub fn slice<Idx>(&self, index: Idx) -> Self
    where
        [T]: Index<Idx, Output = [T]>,
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
        $crate::prelude::DegreeOfFreedom::new($n)
    };
}

/// Find minimum of given values
#[macro_export]
macro_rules! min {
    ($x:expr) => {
        $x
    };
    ($x:expr, $($rest:expr),*$(,)?) => {
        std::cmp::min($x, $crate::min!($($rest),*))
    };
}

/// Find maximum of given values
#[macro_export]
macro_rules! max {
    ($x:expr) => {
        $x
    };
    ($x:expr, $($rest:expr),*$(,)?) => {
        std::cmp::max($x, $crate::max!($($rest),*))
    };
}
