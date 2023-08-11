use derive_new::new;
use ndarray::ArrayView1;
use pcw_fn::{PcwFn, VecPcwFn};
use polyfit_residuals::{
    poly::OwnedNewtonPolynomial, try_fit_poly_with_residual, weighted, PolyFit,
};

use crate::{euclid_sq_metric, DegreeOfFreedom, OrdFloat, SegmentModelSpec, ValidTimeSeriesSample};

/// Models a timeseries via a polynomial function.
#[derive(new, Debug, Eq, PartialEq, Clone)]
pub struct PolynomialApproximator<T>
where
    T: OrdFloat,
{
    dof: DegreeOfFreedom,
    poly: OwnedNewtonPolynomial<T, T>,
    training_error: T,
}

impl<T> PolynomialApproximator<T>
where
    T: OrdFloat,
{
    #[allow(dead_code)]
    pub fn poly(self) -> OwnedNewtonPolynomial<T, T> {
        self.poly
    }

    pub fn fit_to_data(timeseries: &ValidTimeSeriesSample<T>, dof: DegreeOfFreedom) -> Self {
        let PolyFit {
            polynomial,
            residual,
        } = match timeseries.weights() {
            None => try_fit_poly_with_residual(
                ArrayView1::from(timeseries.times()),
                ArrayView1::from(timeseries.response()),
                usize::from(dof) - 1,
            )
            .unwrap(),
            Some(weights) => weighted::try_fit_poly_with_residual(
                ArrayView1::from(timeseries.times()),
                ArrayView1::from(timeseries.response()),
                usize::from(dof) - 1,
                ArrayView1::from(weights),
            )
            .unwrap(),
        };
        PolynomialApproximator::new(dof, polynomial, residual)
    }

    /// Caclculate model's response prediction for a given time
    pub fn prediction(&self, prediction_time: &T) -> T {
        self.poly.left_eval(*prediction_time)
    }
}

/// Compute the full piecewise function mapping time indices to the corresponding models
pub fn full_modelspec_on_seg(
    segment_start_idx: usize,
    segment_stop_idx: usize,
    segment_models: impl IntoIterator<Item = DegreeOfFreedom>,
    cuts: impl IntoIterator<Item = usize>,
) -> VecPcwFn<usize, SegmentModelSpec> {
    let mut models = segment_models.into_iter();
    let (mut cuts, approximations): (Vec<_>, Vec<_>) = cuts
        .into_iter()
        .chain(std::iter::once(segment_stop_idx))
        .scan(segment_start_idx, move |start_idx, cut_idx| {
            let model_spec = SegmentModelSpec {
                start_idx: *start_idx,
                stop_idx: cut_idx,
                seg_dof: models.next().unwrap(),
            };
            *start_idx = cut_idx + 1;
            Some((cut_idx, model_spec))
        })
        .unzip();
    cuts.pop(); // remove the `segment_stop_idx` from the cuts
    VecPcwFn::try_from_iters(cuts, approximations).unwrap()
}

fn approximation_on_segment<T>(
    timeseries: &ValidTimeSeriesSample<T>,
    segment_start_idx: usize,
    segment_stop_idx: usize,
    dof: DegreeOfFreedom,
) -> PolynomialApproximator<T>
where
    T: OrdFloat,
{
    PolynomialApproximator::fit_to_data(
        &timeseries.slice(segment_start_idx..=segment_stop_idx),
        dof,
    )
}

/// Compute the full piecewise function mapping time indices to the corresponding models.
pub fn full_approximation_on_seg<'a, T>(
    timeseries: &'a ValidTimeSeriesSample<'a, T>,
    segment_start_idx: usize,
    segment_stop_idx: usize,
    segment_models: impl IntoIterator<Item = DegreeOfFreedom>,
    cuts: impl IntoIterator<Item = usize>,
) -> VecPcwFn<usize, PolynomialApproximator<T>>
where
    T: OrdFloat,
{
    let mut models = segment_models.into_iter();
    let (mut cuts, approximations): (Vec<_>, Vec<_>) = cuts
        .into_iter()
        .chain(std::iter::once(segment_stop_idx))
        .scan(segment_start_idx, move |start_idx, cut_idx| {
            let a =
                approximation_on_segment(timeseries, *start_idx, cut_idx, models.next().unwrap());
            *start_idx = cut_idx + 1;
            Some((cut_idx, a))
        })
        .unzip();
    cuts.pop(); // remove the `segment_stop_idx` from the cuts
    VecPcwFn::try_from_iters(cuts, approximations).unwrap()
}

/// Returns the error made by this model on `data[0..=segment_stop_idx]` given some sequence
/// of cuts.
pub fn total_training_error<'a, T>(
    timeseries: &'a ValidTimeSeriesSample<'a, T>,
    segment_stop_idx: usize,
    cuts: impl IntoIterator<Item = usize>,
    segment_models: impl IntoIterator<Item = DegreeOfFreedom>,
) -> T
where
    T: OrdFloat,
{
    full_approximation_on_seg(timeseries, 0, segment_stop_idx, segment_models, cuts)
        .into_funcs()
        .map(|seg| seg.training_error)
        .sum()
}

/// Returns the prediction error made by the model from some data interval to the next
/// point after the interval.
pub fn prediction_error<'a, T>(
    timeseries: &'a ValidTimeSeriesSample<'a, T>,
    segment_start_idx: usize,
    segment_stop_idx: usize,
    segment_model: DegreeOfFreedom,
) -> T
where
    T: OrdFloat,
{
    let approx = approximation_on_segment(
        timeseries,
        segment_start_idx,
        segment_stop_idx,
        segment_model,
    );
    (1..=crate::CV_PREDICTION_COUNT)
        .map(|step| {
            euclid_sq_metric(
                &approx.prediction(&timeseries.times()[segment_stop_idx + step]),
                &timeseries.response()[segment_stop_idx + step],
            )
        })
        .sum()
}
