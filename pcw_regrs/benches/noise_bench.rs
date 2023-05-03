use std::num::NonZeroUsize;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use ordered_float::OrderedFloat;
use pcw_regrs::{euclid_sq_metric, PcwApproximator, PcwPolynomialApproximator, TimeSeries};

/*
fn the_func() {
    let time_nn: Vec<_> = sample_times
                .as_array()
                .into_iter()
                .map(|x| -> Option<_> { NotNan::new(*x).ok() })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| PyValueError::new_err("Encountered NaN in times"))?;
    let data_nn: Vec<_> = data
                .as_array()
                .into_iter()
                .map(|x| -> Option<_> { NotNan::new(*x).ok() })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| PyValueError::new_err("Encountered NaN in input"))?;
    let metric = l2sq_metric;

    eprintln!("Beginning error calculation");
    let t1 = Instant::now();
    let approx: PcwPolynomialApproximator<NotNan<f64>, NotNan<f64>, 10> =
        PcwPolynomialApproximator::fit_metric_data_from_model(
            max_seg_dof.map(|dof| NonZeroUsize::new(dof).unwrap()),
            metric,
            TimeSeries::new(&time_nn, &data_nn),
        );
    let t2 = Instant::now();

    solve_jump::dof::models_with_cv_scores(
        max_total_dof.map(|dof| NonZeroUsize::new(dof).unwrap()),
        &approx,
        metric,
    )
}
*/

pub fn criterion_benchmark(c: &mut Criterion) {
    const N_DOFS: usize = 4;
    for data_len in [50, 100, 200, 300, 400, 500] {
        let time = Array1::linspace(OrderedFloat(0.), OrderedFloat(100.), data_len).into_raw_vec();
        let data = Array1::linspace(OrderedFloat(0.), OrderedFloat(200.), data_len).into_raw_vec();
        c.bench_function(
            &format!("error calculation, n={}, dofs≤{}", data_len, N_DOFS),
            |b| {
                b.iter(|| {
                    let timeseries = TimeSeries::new(&time, &data);
                    let approx: PcwPolynomialApproximator<OrderedFloat<f64>> =
                        PcwPolynomialApproximator::fit_metric_data_from_model(
                            black_box(Some(NonZeroUsize::new(N_DOFS).unwrap())),
                            black_box(euclid_sq_metric),
                            black_box(timeseries),
                        );
                    approx
                })
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
