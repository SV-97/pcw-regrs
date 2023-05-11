//! Implementation of the interfaces for (piecewise) constant approximators.
use derive_new::new;
use ndarray::{s, Array1, Array2, ArrayView1};
use num_traits::{real::Real, Signed};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::{fmt::Debug, iter::Sum, num::NonZeroUsize};

use polyfit_residuals::{
    all_residuals, all_residuals_par, poly::OwnedNewtonPolynomial, try_fit_poly_with_residual,
    weighted, PolyFit,
};

use crate::euclid_sq_metric;

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
pub struct PolynomialArgs<'a, TimeData> {
    pub dof: DegreeOfFreedom,
    pub weights: Option<ArrayView1<'a, TimeData>>,
}

impl<'a, T> From<DegreeOfFreedom> for PolynomialArgs<'a, T> {
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
    type Model<'a> = PolynomialArgs<'a, TimeData>;
    fn fit_metric_data_from_model<'a, T: AsRef<[TimeData]>, D: AsRef<[TimeData]>>(
        PolynomialArgs { dof, weights }: Self::Model<'a>,
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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    Error: From<TimeData> + Real, // Clone + FromPrimitive + std::fmt::Display + Num, // + Unsigned,
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
        let xs: ndarray::ArrayBase<ndarray::ViewRepr<&TimeData>, ndarray::Dim<[usize; 1]>> =
            ArrayView1::from_shape(timeseries.len(), timeseries.time()).unwrap();
        let ys = ArrayView1::from_shape(timeseries.len(), timeseries.data()).unwrap();
        let max_degree = max_seg_dof.map(usize::from).unwrap_or(timeseries.len()) + 1;
        // TODO: add feature to switch on off parallelization here
        let mut residuals = match &weights {
            Some(weights) => {
                if cfg!(feature = "parallel_rayon") {
                    weighted::all_residuals_par(xs, ys, max_degree, weights)
                } else {
                    weighted::all_residuals(xs, ys, max_degree, weights)
                }
            }
            None => {
                if cfg!(feature = "parallel_rayon") {
                    all_residuals_par(xs, ys, max_degree)
                } else {
                    all_residuals(xs, ys, max_degree)
                }
            }
        };
        let residuals = if cfg!(feature = "alternate_penalty") {
            let total_time = xs[xs.len() - 1] - xs[0];
            for (seg_start_idx, arr) in residuals.iter_mut().enumerate() {
                for rel_seg_end_idx in 0..arr.shape()[0] {
                    let time_l = xs[seg_start_idx];
                    let time_r = xs[seg_start_idx + rel_seg_end_idx];
                    let t = (time_r - time_l) / total_time;
                    let correction_factor = (t - t.log2());
                    // / TimeData::from(rel_seg_end_idx + 1).unwrap();
                    // maybe_debug::maybe_dbg!(&(
                    //     correction_factor,
                    //     seg_start_idx,
                    //     seg_start_idx + rel_seg_end_idx
                    // ));
                    arr.slice_mut(s![rel_seg_end_idx, ..])
                        .map_inplace(|x: &mut TimeData| *x = *x * correction_factor);
                }
            }
            residuals
        } else {
            residuals
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
        PolynomialArgs { dof, weights }: PolynomialArgs<TimeData>,
        mut _metric: impl FnMut(&TimeData, &TimeData) -> Error,
    ) -> Error {
        // The weights for the segment up to and including the prediction point
        let segment_weights = match (&weights, self.args.weights.as_ref()) {
            (Some(weights), _) => Some(weights.view()),
            (_, Some(weights)) => Some(weights.slice(s![segment_start_idx..=segment_stop_idx])),
            (_, _) => None,
        };
        // The polynomial approximating this segment
        let a = match segment_weights {
            Some(w) => self.approximation_on_segment(
                segment_start_idx,
                segment_stop_idx,
                PolynomialArgs {
                    dof,
                    weights: Some(w.slice(s![..w.len()])),
                },
            ),
            None => self.approximation_on_segment(
                segment_start_idx,
                segment_stop_idx,
                PolynomialArgs { dof, weights: None },
            ),
        };
        // The weight for the point which we predict
        let next_w = weights
            .map(|w| From::from(*w.last().unwrap()))
            .unwrap_or_else(Error::one);
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
        next_w * From::from(euclid_sq_metric(&predicted_y, next_y))
    }
}
