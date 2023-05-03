//! Implementation of the interfaces for (piecewise) constant approximators.
use derive_new::new;

use ndarray::{s, Array1, Array2, ArrayView1};
use num_traits::{real::Real, Signed};

use std::{fmt::Debug, iter::Sum, num::NonZeroUsize};

use polyfit_residuals::{
    all_residuals_par, poly::OwnedNewtonPolynomial, try_fit_poly_with_residual, weighted, PolyFit,
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

#[derive(new, Debug, Eq, PartialEq, Clone)]
pub struct PolynomialArgs<TimeData> {
    pub dof: DegreeOfFreedom,
    pub weights: Option<Array1<TimeData>>,
}

impl<T> From<DegreeOfFreedom> for PolynomialArgs<T> {
    fn from(dof: DegreeOfFreedom) -> Self {
        Self { dof, weights: None }
    }
}

impl<TimeData, Error> ErrorApproximator<TimeData, TimeData, Error>
    for PolynomialApproximator<TimeData, Error>
where
    TimeData: Real + Signed + Sum + Eq + 'static,
    Error: Clone + Real + From<TimeData>, // + Unsigned,
{
    type Model = PolynomialArgs<TimeData>;
    fn fit_metric_data_from_model<T: AsRef<[TimeData]>, D: AsRef<[TimeData]>>(
        PolynomialArgs { dof, weights }: Self::Model,
        _metric: impl FnMut(&TimeData, &TimeData) -> Error,
        data: TimeSeries<TimeData, TimeData, T, D>,
    ) -> Self {
        let PolyFit {
            polynomial,
            residual,
        } = match weights {
            None => try_fit_poly_with_residual(
                ArrayView1::from_shape(data.len(), data.time()).unwrap(),
                ArrayView1::from_shape(data.len(), data.data()).unwrap(),
                usize::from(dof) - 1,
            )
            .unwrap(),
            Some(weights) => weighted::try_fit_poly_with_residual(
                ArrayView1::from_shape(data.len(), data.time()).unwrap(),
                ArrayView1::from_shape(data.len(), data.data()).unwrap(),
                usize::from(dof) - 1,
                weights.view(),
            )
            .unwrap(),
        };
        PolynomialApproximator::new(dof, polynomial, <Error as From<TimeData>>::from(residual))
    }

    fn training_error(&self) -> Error {
        self.training_error
    }

    fn prediction(&self, prediction_time: &TimeData) -> TimeData {
        self.poly.left_eval(*prediction_time)
    }

    // fn model(&self) -> &Self::Model {
    //     &self.dof
    // }
}

#[derive(new, Debug, Eq, PartialEq, Clone)]
pub struct PcwPolynomialArgs<TimeData> {
    /// Maximal degrees of freedom of polynomials considered
    pub max_seg_dof: Option<DegreeOfFreedom>,
    /// Weights for a weighted regression
    pub weights: Option<Array1<TimeData>>,
}

impl<TimeData> Default for PcwPolynomialArgs<TimeData> {
    fn default() -> Self {
        PcwPolynomialArgs {
            max_seg_dof: Some(unsafe { NonZeroUsize::new_unchecked(10) }),
            weights: None,
        }
    }
}

/// Models a timeseries via a piecewise polynomial function.
#[derive(Debug)]
pub struct PcwPolynomialApproximator<TimeData> {
    args: PcwPolynomialArgs<TimeData>,
    times: Array1<TimeData>,
    data: Array1<TimeData>,
    residuals: Vec<Array2<TimeData>>,
}

/*
impl<TimeData> PcwPolynomialApproximator<TimeData> {
    fn max_seg_dof(&self) -> Option<DegreeOfFreedom> {
        self.args.max_seg_dof
    }
}
*/

impl<TimeData, Error>
    PcwApproximator<PolynomialApproximator<TimeData, Error>, TimeData, TimeData, Error>
    for PcwPolynomialApproximator<TimeData>
where
    TimeData: Real + Signed + Sum + Eq + 'static + Send + Sync,
    Error: From<TimeData> + Clone + Real, // Clone + FromPrimitive + std::fmt::Display + Num, // + Unsigned,
{
    /// Maximal degrees of freedom of polynomials considered
    type Model = PcwPolynomialArgs<TimeData>;

    fn fit_metric_data_from_model<T: AsRef<[TimeData]>, D: AsRef<[TimeData]>>(
        PcwPolynomialArgs {
            max_seg_dof,
            weights,
        }: Self::Model,
        _metric: impl FnMut(&TimeData, &TimeData) -> Error,
        timeseries: TimeSeries<TimeData, TimeData, T, D>,
    ) -> Self {
        assert!(!timeseries.is_empty());
        let xs = ArrayView1::from_shape(timeseries.len(), timeseries.time()).unwrap();
        let ys = ArrayView1::from_shape(timeseries.len(), timeseries.data()).unwrap();
        let max_degree = max_seg_dof.map(usize::from).unwrap_or(timeseries.len()) + 1;
        // TODO: add feature to switch on off parallelization here
        let residuals = match &weights {
            Some(weights) => weighted::all_residuals_par(xs, ys, max_degree, weights),
            None => all_residuals_par(xs, ys, max_degree),
        };
        let args = PcwPolynomialArgs {
            max_seg_dof,
            weights,
        };
        Self {
            args,
            times: timeseries.time().iter().cloned().collect(),
            data: timeseries.data().iter().cloned().collect(),
            residuals,
        }
    }

    fn approximation_on_segment(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
        segment_args: PolynomialArgs<TimeData>,
    ) -> PolynomialApproximator<TimeData, Error> {
        PolynomialApproximator::fit_metric_data_from_model(
            segment_args,
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
        PolynomialArgs { dof, .. }: PolynomialArgs<TimeData>,
        _metric: impl FnMut(&TimeData, &TimeData) -> Error,
    ) -> Error {
        <Error as From<TimeData>>::from(
            self.residuals[segment_start_idx]
                [[segment_stop_idx - segment_start_idx, usize::from(dof) - 1]],
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

    fn model(&self) -> &Self::Model {
        &self.args
    }

    /// Returns the prediction error made by the model from some data interval to the next
    /// point after the interval.
    fn prediction_error(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
        PolynomialArgs { dof, .. }: PolynomialArgs<TimeData>,
        mut metric: impl FnMut(&TimeData, &TimeData) -> Error,
    ) -> Error {
        let w = self
            .args
            .weights
            .as_ref()
            .map(|w| From::from(w[segment_stop_idx + 1]))
            .unwrap_or_else(Error::one);
        let a = self.approximation_on_segment(
            segment_start_idx,
            segment_stop_idx,
            PolynomialArgs {
                dof,
                weights: self.args.weights.as_ref().map(|w| {
                    let slice_weights = w.slice(s![segment_start_idx..=segment_stop_idx]);
                    slice_weights.to_owned()
                }),
            },
        );
        let next_t: &TimeData = <PcwPolynomialApproximator<TimeData> as PcwApproximator<
            PolynomialApproximator<TimeData, Error>,
            TimeData,
            TimeData,
            Error,
        >>::time_at(self, segment_stop_idx + 1);
        let next_y = <PcwPolynomialApproximator<TimeData> as PcwApproximator<
            PolynomialApproximator<TimeData, Error>,
            TimeData,
            TimeData,
            Error,
        >>::data_at(self, segment_stop_idx + 1);
        let predicted_y = <PolynomialApproximator<TimeData, _> as ErrorApproximator<
            TimeData,
            TimeData,
            Error,
        >>::prediction(&a, next_t);
        w * metric(&predicted_y, next_y)
    }
}
