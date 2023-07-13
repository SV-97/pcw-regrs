//! Implementation of the interfaces for (piecewise) constant approximators.
use super::{ErrorApproximator, PcwApproximator, TimeSeries};
use derive_new::new;

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{FromPrimitive, Num, Signed};

use std::fmt::Debug;

// A constant approximator takes no `Time` typearg since the model is invariant to
// (inhomogenous) rescalings of the time axis.
/// Models a timeseries via a constant function.
#[derive(new, Debug, Eq, PartialEq, Clone, Copy)]
pub struct ConstantApproximator<Data, Error> {
    mean: Data,
    approximation_error: Error,
}

impl<Data, Error> ConstantApproximator<Data, Error> {
    pub fn mean(self) -> Data {
        self.mean
    }
}

impl<Time, Data, Error> ErrorApproximator<Time, Data, Error> for ConstantApproximator<Data, Error>
where
    Data: Clone + FromPrimitive + Signed,
    Error: Clone + FromPrimitive + Num, // + Unsigned,
{
    type Model = ();
    fn fit_metric_data_from_model<T: AsRef<[Time]>, D: AsRef<[Data]>>(
        _: Self::Model,
        mut metric: impl FnMut(&Data, &Data) -> Error,
        data: TimeSeries<Time, Data, T, D>,
    ) -> Self {
        let v = ArrayView1::from(data.data());
        let mean = v.mean().unwrap();
        let approximation_error = v.map(|x| metric(x, &mean)).mean().unwrap();
        Self::new(mean, approximation_error)
    }

    fn training_error(&self) -> Error {
        self.approximation_error.clone()
    }

    fn prediction(&self, _prediction_time: &Time) -> Data {
        self.mean.clone()
    }

    // fn model(&self) -> &Self::Model {
    //     &()
    // }
}

/// Models a timeseries via a piecewise constant function.
pub struct PcwConstantApproximator<Time, Data, Error> {
    data: Array1<Data>,
    times: Array1<Time>,
    approximations: Array2<Option<ConstantApproximator<Data, Error>>>,
}

impl<Time, Data, Error> PcwApproximator<ConstantApproximator<Data, Error>, Time, Data, Error>
    for PcwConstantApproximator<Time, Data, Error>
where
    Time: Clone,
    Data: Clone + FromPrimitive + Signed,
    Error: Clone + FromPrimitive + Num, // + Unsigned,
{
    type Model = ();
    fn fit_metric_data_from_model<T: AsRef<[Time]>, D: AsRef<[Data]>>(
        _base_model: Self::Model,
        mut metric: impl FnMut(&Data, &Data) -> Error,
        timeseries: TimeSeries<Time, Data, T, D>,
    ) -> Self {
        // TODO: make this more efficient
        let approximations =
            Array2::from_shape_fn((timeseries.len(), timeseries.len()), |(l, r)| {
                if l > r {
                    None
                } else {
                    let data_slice = ArrayView1::from(&timeseries.data()[l..=r]);
                    let mean = data_slice.mean().unwrap();
                    // TODO: maybe have to square the metric here
                    let error = data_slice.map(|d| metric(d, &mean)).sum();
                    Some(ConstantApproximator::new(mean, error))
                }
            });
        // println!(
        //     "Errors = \n{:.2}\n",
        //     approximations.map(|x| x
        //         .clone()
        //         .map(|y| y.approximation_error.clone())
        //         .unwrap_or(Error::zero()))
        // );
        // println!("{:?}", approximations);
        Self {
            data: timeseries.data().iter().cloned().collect(),
            times: timeseries.time().iter().cloned().collect(),
            approximations,
        }
    }

    fn approximation_on_segment(
        &self,
        segment_start_idx: usize,
        segment_stop_idx: usize,
        _segment_model: (),
    ) -> ConstantApproximator<Data, Error> {
        self.approximations[[segment_start_idx, segment_stop_idx]]
            .clone()
            .unwrap()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }

    fn data_at(&self, idx: usize) -> &Data {
        &self.data[idx]
    }

    fn time_at(&self, idx: usize) -> &Time {
        &self.times[idx]
    }

    fn model(&self) -> &Self::Model {
        &()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::NotNan;

    #[test]
    fn pcw_const() {
        let n = |x| NotNan::new(x).unwrap();
        let raw_data = vec![2., 1.9, 1., 1.1];
        let data: Vec<_> = raw_data.clone().into_iter().map(n).collect();
        let metric = |x: &NotNan<f64>, y: &NotNan<f64>| {
            let abs = (x - y).abs();
            abs * abs // NonNegative::new(
        };
        let approx: PcwConstantApproximator<NotNan<f64>, NotNan<f64>, NotNan<f64>> =
            PcwConstantApproximator::fit_metric_data_from_model(
                (),
                metric,
                TimeSeries::<NotNan<f64>, _, _, _>::with_homogenous_time(data),
            );
        assert_eq!(4, approx.data_len());
        assert_eq!(&n(raw_data[0]), approx.data_at(0));
        assert_eq!(&n(raw_data[1]), approx.data_at(1));
        assert_eq!(&n(raw_data[2]), approx.data_at(2));
        assert_eq!(&n(raw_data[3]), approx.data_at(3));
        // Test values in the following are calculated "manually" using numpy
        // These tests may need to be switched over to approximate equalities
        // if the internal algorithm of the constant approximator is changed.
        assert_eq!(
            ConstantApproximator::new(n(1.5), n(0.8199999999999998)),
            approx.approximation_on_segment(0, 3, ())
        );
        assert_eq!(
            ConstantApproximator::new(n(1.3333333333333333), n(0.48666666666666647)),
            approx.approximation_on_segment(1, 3, ())
        );
        assert_eq!(
            ConstantApproximator::new(n(1.05), n(0.005000000000000009)),
            approx.approximation_on_segment(2, 3, ())
        );
        assert_eq!(
            ConstantApproximator::new(n(1.1), n(0.0)),
            approx.approximation_on_segment(3, 3, ())
        );

        assert_eq!(
            ConstantApproximator::new(n(1.6333333333333335), n(0.6066666666666667)),
            approx.approximation_on_segment(0, 2, ())
        );
        assert_eq!(
            ConstantApproximator::new(n(1.45), n(0.4049999999999999)),
            approx.approximation_on_segment(1, 2, ())
        );
        assert_eq!(
            ConstantApproximator::new(n(1.0), n(0.0)),
            approx.approximation_on_segment(2, 2, ())
        );

        assert_eq!(
            ConstantApproximator::new(n(1.95), n(0.005000000000000009)),
            approx.approximation_on_segment(0, 1, ())
        );
        assert_eq!(
            ConstantApproximator::new(n(1.9), n(0.0)),
            approx.approximation_on_segment(1, 1, ())
        );

        assert_eq!(
            ConstantApproximator::new(n(2.0), n(0.0)),
            approx.approximation_on_segment(0, 0, ())
        );
    }
}
