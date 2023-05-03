//! Implementation of the interfaces for (piecewise) constant approximators.
use derive_new::new;

use ndarray::{s, Array1, Array2, ArrayView1};
use num_traits::{real::Real, Signed};

use std::{fmt::Debug, iter::Sum, num::NonZeroUsize};

use polyfit_residuals::{
    all_residuals, poly::OwnedNewtonPolynomial, try_fit_poly_with_residual, PolyFit,
};

use super::{DegreeOfFreedom, ErrorApproximator, PcwApproximator, TimeSeries};

/// Models a timeseries via a polynomial function.
#[derive(new, Debug, Eq, PartialEq, Clone)]
pub struct PolynomialApproximator<Data, Error> {
    dof: DegreeOfFreedom,
    poly: OwnedNewtonPolynomial<Data, Data>,
    training_error: Error,
}
impl<D, E> PolynomialApproximator<D, E> {
    pub fn poly(self) -> OwnedNewtonPolynomial<D, D> {
        self.poly
    }
}

impl<TimeData, Error> ErrorApproximator<TimeData, TimeData, Error>
    for PolynomialApproximator<TimeData, Error>
where
    TimeData: Real + Signed + Sum + Eq + 'static,
    Error: Clone + Real + From<TimeData>, // + Unsigned,
{
    type Model = DegreeOfFreedom;
    fn fit_metric_data_from_model<T: AsRef<[TimeData]>, D: AsRef<[TimeData]>>(
        dof: Self::Model,
        _metric: impl FnMut(&TimeData, &TimeData) -> Error,
        data: TimeSeries<TimeData, TimeData, T, D>,
    ) -> Self {
        let PolyFit {
            polynomial,
            residual,
        } = try_fit_poly_with_residual(
            ArrayView1::from_shape(data.len(), data.time()).unwrap(),
            ArrayView1::from_shape(data.len(), data.data()).unwrap(),
            usize::from(dof) - 1,
        )
        .unwrap();
        PolynomialApproximator::new(dof, polynomial, <Error as From<TimeData>>::from(residual))
    }

    fn training_error(&self) -> Error {
        self.training_error
    }

    fn prediction(&self, prediction_time: &TimeData) -> TimeData {
        self.poly.left_eval(*prediction_time)
    }

    fn model(&self) -> &Self::Model {
        &self.dof
    }
}

/// Models a timeseries via a piecewise polynomial function.
#[derive(Debug)]
pub struct PcwPolynomialApproximator<TimeData> {
    max_seg_dof: Option<DegreeOfFreedom>,
    times: Array1<TimeData>,
    data: Array1<TimeData>,
    residuals: Vec<Array2<TimeData>>,
}

impl<TimeData, Error>
    PcwApproximator<PolynomialApproximator<TimeData, Error>, TimeData, TimeData, Error>
    for PcwPolynomialApproximator<TimeData>
where
    TimeData: Real + Signed + Sum + Eq + 'static + Send + Sync,
    Error: From<TimeData> + Clone + Real, // Clone + FromPrimitive + std::fmt::Display + Num, // + Unsigned,
{
    /// Maximal degrees of freedom of polynomials considered
    type Model = Option<DegreeOfFreedom>;

    fn fit_metric_data_from_model<T: AsRef<[TimeData]>, D: AsRef<[TimeData]>>(
        max_seg_dof: Self::Model,
        _metric: impl FnMut(&TimeData, &TimeData) -> Error,
        timeseries: TimeSeries<TimeData, TimeData, T, D>,
    ) -> Self {
        assert!(!timeseries.is_empty());
        let xs = ArrayView1::from_shape(timeseries.len(), timeseries.time()).unwrap();
        let ys = ArrayView1::from_shape(timeseries.len(), timeseries.data()).unwrap();
        let residuals = all_residuals(
            xs,
            ys,
            max_seg_dof.map(usize::from).unwrap_or(timeseries.len()) + 1,
        );
        Self {
            max_seg_dof,
            times: timeseries.time().iter().cloned().collect(),
            data: timeseries.data().iter().cloned().collect(),
            residuals,
        }
    }

    fn approximation_on_segment(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
        segment_dof: NonZeroUsize,
    ) -> PolynomialApproximator<TimeData, Error> {
        PolynomialApproximator::fit_metric_data_from_model(
            segment_dof,
            |_, _| panic!("Called metric that shouldn't be called"),
            TimeSeries::new(
                self.times
                    .slice(s![segment_start_idx..=segment_stop_idx])
                    .as_slice()
                    .unwrap(),
                self.data
                    .slice(s![segment_start_idx..=segment_stop_idx])
                    .as_slice()
                    .unwrap(),
            ),
        )
    }

    fn training_error(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
        segment_model: NonZeroUsize,
        _metric: impl FnMut(&TimeData, &TimeData) -> Error,
    ) -> Error {
        <Error as From<TimeData>>::from(
            self.residuals[segment_start_idx][[
                segment_stop_idx - segment_start_idx,
                usize::from(segment_model) - 1,
            ]],
        )
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }

    fn data_at(&self, idx: usize) -> &TimeData {
        &self.data[idx]
    }

    fn time_at(&self, idx: usize) -> &TimeData {
        &self.times[idx]
    }

    /// Maximum degree of freedom on each segment of data
    fn model(&self) -> &Self::Model {
        &self.max_seg_dof
    }
}
