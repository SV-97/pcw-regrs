//! The general abstract interfaces used by the solver as well as some concrete implementations.
use is_sorted::IsSorted;
use itertools::Itertools;
use num_traits::FromPrimitive;
use pcw_fn::{PcwFn, VecPcwFn};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::{marker::PhantomData, ops::Index};

mod constant;
mod polynomial;

pub use constant::*;
pub use polynomial::*;

/// A discrete measure of model complexity - lower number means lower complexity.
pub type DegreeOfFreedom = NonZeroUsize;
/// A sequence of data consisting of pairs of "sample times" and corresponding "measurements" or "response values".
pub struct TimeSeries<Time, Data, T, D>
where
    T: AsRef<[Time]>,
    D: AsRef<[Data]>,
{
    time: T,
    data: D,
    _phantom: PhantomData<(Time, Data)>,
}

impl<Time, Data, D: AsRef<[Data]>> TimeSeries<Time, Data, Vec<Time>, D> {
    /// Create a new timeseries just from datapoints by assuming that the sample times
    /// are equidistantly spaced.
    pub fn with_homogenous_time(data: D) -> Self
    where
        Time: FromPrimitive + PartialOrd,
    {
        let times = (0..data.as_ref().len())
            .into_iter()
            .map(Time::from_usize)
            .map(Option::unwrap)
            .collect_vec();
        TimeSeries::new(times, data)
    }
}

impl<Time, Data, T: AsRef<[Time]>, D: AsRef<[Data]>> TimeSeries<Time, Data, T, D> {
    /// Construct a new time series from its constituent parts
    ///
    /// # Panics
    /// Panics if the inputs have differents lengths (as slices).
    pub fn new(time: T, data: D) -> Self
    where
        Time: PartialOrd,
    {
        assert_eq!(time.as_ref().len(), data.as_ref().len());
        assert!(IsSorted::is_sorted_by(
            &mut time.as_ref().iter(),
            crate::strict_partial_cmp
        ));
        // TODO: maybe assert strict order on time
        TimeSeries {
            time,
            data,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn time(&self) -> &[Time] {
        self.time.as_ref()
    }

    #[inline]
    pub fn data(&self) -> &[Data] {
        self.data.as_ref()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.time().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.time().is_empty()
    }

    /// Obtain a new timeseries by slicing out a piece of another one immutably.
    pub fn slice<I>(&self, idx: I) -> TimeSeries<Time, Data, &[Time], &[Data]>
    where
        I: Clone,
        [Time]: Index<I, Output = [Time]>,
        [Data]: Index<I, Output = [Data]>,
    {
        TimeSeries {
            time: &self.time()[idx.clone()],
            data: &self.data()[idx],
            _phantom: PhantomData,
        }
    }
}

/// Local model for a timeseries. This trait is to be used in conjunction with instances
/// of [PcwApproximator].
pub trait ErrorApproximator<Time, Data, Error> {
    /// The base model of the approximation with additional model coefficients (i.e. degree
    /// for a polynomial model).
    type Model;

    /// A way to fit the generic base model to specific data subject to some metric on the data
    fn fit_metric_data_from_model<T: AsRef<[Time]>, D: AsRef<[Data]>>(
        base_model: Self::Model,
        metric: impl FnMut(&Data, &Data) -> Error,
        data: TimeSeries<Time, Data, T, D>,
    ) -> Self;

    /// The training error of the model
    fn training_error(&self) -> Error;

    /// What the model predicts for some input index (= a point in time)
    fn prediction(&self, prediction_time: &Time) -> Data;

    /// The base model of the approximator
    fn model(&self) -> &Self::Model;
}

/// Compute the full piecewise function mapping time indices to the corresponding models.
pub fn full_approximation_on_seg<'a, U, P, SegmentApprox, Time, Data, Error>(
    approx: &P,
    segment_start_idx: usize,
    segment_stop_idx: usize,
    segment_models: impl IntoIterator<Item = SegmentApprox::Model>,
    cuts: impl IntoIterator<Item = U>,
) -> VecPcwFn<usize, SegmentApprox>
where
    U: Into<maybe_owned::MaybeOwned<'a, usize>>,
    P: PcwApproximator<SegmentApprox, Time, Data, Error> + ?Sized,
    SegmentApprox: ErrorApproximator<Time, Data, Error>,
    SegmentApprox::Model: Clone,
{
    let mut models = segment_models.into_iter();
    let (mut cuts, approximations): (Vec<_>, Vec<_>) = cuts
        .into_iter()
        .map(|cut_idx| *cut_idx.into())
        .chain(std::iter::once(segment_stop_idx))
        .scan(segment_start_idx, move |start_idx, cut_idx| {
            let a = approx.approximation_on_segment(*start_idx, cut_idx, models.next().unwrap());
            *start_idx = cut_idx + 1;
            Some((cut_idx, a))
        })
        .unzip();
    cuts.pop(); // remove the `segment_stop_idx` from the cuts
    VecPcwFn::try_from_iters(cuts, approximations).unwrap()
}

/// A specification of a model on some segment of data given by the start and stop indices w.r.t. a timeseries.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SegmentModelSpec<T> {
    /// The index of the point in time (in the original timeseries) at which this segment starts.
    pub start_idx: usize,
    /// The index of the point in time (in the original timeseries) at which this segment stops.
    pub stop_idx: usize,
    /// The model for the segment.
    pub model: T,
}

/// Compute the full piecewise function mapping time indices to the corresponding models
pub fn full_modelspec_on_seg<'a, U, P, SegmentApprox, Time, Data, Error>(
    _approx: &P,
    segment_start_idx: usize,
    segment_stop_idx: usize,
    segment_models: impl IntoIterator<Item = SegmentApprox::Model>,
    cuts: impl IntoIterator<Item = U>,
) -> VecPcwFn<usize, SegmentModelSpec<SegmentApprox::Model>>
where
    U: Into<maybe_owned::MaybeOwned<'a, usize>>,
    P: PcwApproximator<SegmentApprox, Time, Data, Error> + ?Sized,
    SegmentApprox: ErrorApproximator<Time, Data, Error>,
    SegmentApprox::Model: Clone,
{
    let mut models = segment_models.into_iter();
    let (mut cuts, approximations): (Vec<_>, Vec<_>) = cuts
        .into_iter()
        .map(|cut_idx| *cut_idx.into())
        .chain(std::iter::once(segment_stop_idx))
        .scan(segment_start_idx, move |start_idx, cut_idx| {
            let model_spec = SegmentModelSpec {
                start_idx: *start_idx,
                stop_idx: cut_idx,
                model: models.next().unwrap(),
            };
            *start_idx = cut_idx + 1;
            Some((cut_idx, model_spec))
        })
        .unzip();
    cuts.pop(); // remove the `segment_stop_idx` from the cuts
    VecPcwFn::try_from_iters(cuts, approximations).unwrap()
}

/// Types implementing this trait effectively define a model space Ω and the corresponding model
/// fitting functions and provide an interface for interacting with those models.
pub trait PcwApproximator<SegmentApprox, Time, Data, Error>
where
    SegmentApprox: ErrorApproximator<Time, Data, Error>,
{
    /// The base model of the approximation with additional model coefficients (i.e. maximal degree
    /// for a polynomial model). This essentially allows for hyperparameters for Ω and the model-
    /// fitting functions.
    type Model;

    /// A way to fit the generic base model to specific data subject to some metric on the data.
    fn fit_metric_data_from_model<T: AsRef<[Time]>, D: AsRef<[Data]>>(
        base_model: Self::Model,
        metric: impl FnMut(&Data, &Data) -> Error,
        data: TimeSeries<Time, Data, T, D>,
    ) -> Self;

    /// # Returns
    /// Length of data the approximator was constructed with.
    fn data_len(&self) -> usize;

    /// # Returns
    /// Approximation on `data[segment_start_idx..=segment_stop_idx]`.
    ///
    /// # Panics
    /// May panic if indices are outside the data domain.
    fn approximation_on_segment(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
        segment_model: SegmentApprox::Model,
    ) -> SegmentApprox;

    /// Provides access to underlying sample times of the timeseries.
    fn time_at(&self, idx: usize) -> &Time;

    /// Provides access to underlying data / response of the timeseries.
    fn data_at(&self, idx: usize) -> &Data;

    /// Returns the error made by this model on `data[0..=segment_stop_idx]` given some sequence
    /// of cuts.
    fn total_training_error<'a, U>(
        &self,
        segment_stop_idx: usize,
        cuts: impl IntoIterator<Item = U>,
        segment_models: impl IntoIterator<Item = SegmentApprox::Model>,
    ) -> Error
    where
        U: Into<maybe_owned::MaybeOwned<'a, usize>>,
        Error: std::iter::Sum, // Zero + AddAssign,
        SegmentApprox::Model: Clone,
    {
        full_approximation_on_seg(self, 0, segment_stop_idx, segment_models, cuts)
            .into_funcs()
            .map(|seg| seg.training_error())
            .sum()
    }

    /// Returns the residual training error made by the model on some data interval.
    fn training_error(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
        segment_model: SegmentApprox::Model,
        mut metric: impl FnMut(&Data, &Data) -> Error,
    ) -> Error {
        metric(
            &self
                .approximation_on_segment(segment_start_idx, segment_stop_idx, segment_model)
                .prediction(self.time_at(segment_stop_idx)),
            self.data_at(segment_stop_idx),
        )
    }

    /// Returns the prediction error made by the model from some data interval to the next
    /// point after the interval.
    fn prediction_error(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
        segment_model: SegmentApprox::Model,
        mut metric: impl FnMut(&Data, &Data) -> Error,
    ) -> Error {
        metric(
            &self
                .approximation_on_segment(segment_start_idx, segment_stop_idx, segment_model)
                .prediction(self.time_at(segment_stop_idx + 1)),
            self.data_at(segment_stop_idx + 1),
        )
    }

    /// The underlying "hypermodel" used when constructing this approximator.
    fn model(&self) -> &Self::Model;
}
