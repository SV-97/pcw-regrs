use std::{num::NonZeroUsize, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{concatenate, Array, Axis};
use ndarray_rand::RandomExt;
use ordered_float::OrderedFloat;
use pcw_regrs::{try_fit_pcw_poly, TimeSeriesSample, UserParams};
use rand_distr::Normal;

pub fn bench_try_fit_pcw_poly(c: &mut Criterion) {
    const N: usize = 700;
    let t0 = Array::linspace(0., 0.3, N / 3);
    let dt = t0[1] - t0[0];
    let t1 = Array::linspace(0.3 + dt, 0.6, N / 3);
    let t2 = Array::linspace(0.6 + dt, 1., N - 2 * (N / 3));

    let ts = concatenate!(Axis(0), t0.clone(), t1.clone(), t2.clone());
    let noise = Array::random(ts.shape(), Normal::new(0., 0.05).unwrap());
    let ys = concatenate!(
        Axis(0),
        t0.clone() * t0,
        t1,
        t2.mapv(|t: f64| 0.7 - t.powi(3))
    ) + noise;

    let sample =
        TimeSeriesSample::try_new(ts.as_slice().unwrap(), ys.as_slice().unwrap(), None).unwrap();
    let params = UserParams {
        max_seg_dof: NonZeroUsize::new(10),
        max_total_dof: NonZeroUsize::new(200),
    };

    c.bench_function(&format!("Fit {}", N), |b| {
        b.iter(|| {
            let _ = try_fit_pcw_poly::<OrderedFloat<f64>>(black_box(&sample), black_box(&params));
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(60));
    targets = bench_try_fit_pcw_poly
);
criterion_main!(benches);
