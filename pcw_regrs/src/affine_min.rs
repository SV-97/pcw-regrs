//! This module implements an efficient algorithm for determining the pointwise minimum of
//! a collection of affine linear functions on ℝ.

use derive_new::new;
use num_traits::{Num, Zero};
use std::borrow::Borrow;

use std::fmt::Debug;
use std::ops::{Add, Sub};

use pcw_fn::PcwFn;

/// An affine line function given by it's `slope` and `intercept`.
#[derive(new, Clone, Copy, Eq, PartialEq, Debug, Default)]
pub struct AffineFunction<T> {
    pub slope: T,
    pub intercept: T,
}

impl<T: Num + Clone> AffineFunction<T> {
    /// Evaluate [self] at a point `x` of its domain.
    ///
    /// # Note
    /// This may be deprecated in favour of a [std::ops::Fn] impl once
    /// (fn_traits [#29625](https://github.com/rust-lang/rust/issues/29625)) is stabilized.
    #[inline]
    pub fn evaluate_at(&self, x: T) -> T {
        self.slope.clone() * x + self.intercept.clone()
    }

    /// Calculate the point x where `self.evaluate_at(x) == other.evaluate_at(x)` if it exists.
    #[inline]
    pub fn graph_intersection(&self, other: &Self) -> Option<T> {
        if self.slope == other.slope {
            if self.intercept == other.intercept {
                Some(T::zero())
            } else {
                None
            }
        } else {
            Some(
                (other.intercept.clone() - self.intercept.clone())
                    / (self.slope.clone() - other.slope.clone()),
            )
        }
    }
}

impl<T: Add> Add for AffineFunction<T> {
    type Output = AffineFunction<<T as Add>::Output>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        AffineFunction {
            slope: self.slope + rhs.slope,
            intercept: self.intercept + rhs.intercept,
        }
    }
}

impl<T: Sub> Sub for AffineFunction<T> {
    type Output = AffineFunction<<T as Sub>::Output>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        AffineFunction {
            slope: self.slope - rhs.slope,
            intercept: self.intercept - rhs.intercept,
        }
    }
}

impl<T: Zero> AffineFunction<T> {
    #[allow(unused)]
    fn zero() -> Self {
        AffineFunction {
            slope: T::zero(),
            intercept: T::zero(),
        }
    }
}

/// Sort a collection of affine functions in the correct order needed for finding the
/// pointwise minimum using [pointwise_minimum_of_affines].
#[allow(unused)]
pub fn sort_affines_for_minimization<T: Ord>(affines: &mut [AffineFunction<T>]) {
    affines.sort_unstable_by(|f, g| f.slope.cmp(&g.slope).reverse());
}

/// Find pointwise minimum of a collection of **pairwise nonparallel** affine functions.
///
/// # Args
/// * `affines` - The collection of [AffineFunction]s to consider. Note that this **has** to be
///     *strictly* sorted by by [AffineFunction::slope] in *descending* order.
///
/// # Panics
/// Panics if two lines are parallel or if the input is not properly sorted. Note
/// that failing the non-parallelity precondition induces a failure in the order precondition,
/// since in this case there can be no strict order.
pub fn pointwise_minimum_of_affines<T, S, I, Pcw>(mut affines: I) -> Option<Pcw>
where
    I: ExactSizeIterator + Iterator<Item = S>,
    S: Borrow<AffineFunction<T>> + Default,
    T: Ord + Clone + Num + Default,
    Pcw: PcwFn<T, S>,
{
    match affines.len() {
        0 => None,                                       // An empty set has no minimum
        1 => Some(Pcw::global(affines.next().unwrap())), // Minium of a set with one element is that element
        _ => {
            let mut minimals: Vec<S> = vec![affines.next().unwrap(), affines.next().unwrap()];
            let mut jumps = vec![minimals[0]
                .borrow()
                .graph_intersection(minimals[1].borrow())
                .unwrap()]; // expect("No valid intersection between lines: lines are parallel.")
            for right_fn in affines {
                let mut left_fn: &S = minimals.last().unwrap();
                while {
                    if let Some(a1) = jumps.last() {
                        // Note that we include the `==` case in our pop condition here. This prevents
                        // us from including the same jump location multiple times for different functions.
                        (right_fn.borrow().clone() - left_fn.borrow().clone())
                            .evaluate_at((a1 as &T).clone())
                            <= T::zero()
                    } else {
                        false
                    }
                } {
                    minimals.pop();
                    jumps.pop();
                    left_fn = minimals.last().unwrap();
                }
                let new_jump = match left_fn.borrow().graph_intersection(right_fn.borrow()) {
                    Some(x) => x,
                    None => {
                        // dbg!((&left_fn, &right_fn));
                        panic!("Failed to compute point of intersection.")
                    }
                };
                while {
                    if let Some(j) = jumps.last() {
                        &new_jump <= j
                    } else {
                        false
                    }
                } {
                    jumps.pop();
                    minimals.pop();
                }
                jumps.push(new_jump);
                minimals.push(right_fn);
            }
            Some(Pcw::try_from_iters(jumps, minimals).unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::NotNan;
    use pcw_fn::VecPcwFn;

    #[test]
    fn identity_on_single_func() {
        let f = AffineFunction::new(NotNan::new(0.0).unwrap(), NotNan::new(1.0).unwrap());
        let x = [f];
        assert_eq!(
            VecPcwFn::global(f),
            pointwise_minimum_of_affines(x.clone().into_iter()).unwrap(),
        )
    }

    #[test]
    fn correct_single_intersect() {
        let f = AffineFunction::new(NotNan::new(1.0).unwrap(), NotNan::new(0.0).unwrap());
        let g = AffineFunction::new(NotNan::new(-1.0).unwrap(), NotNan::new(2.0).unwrap());
        let mut x = [f, g];
        sort_affines_for_minimization(&mut x);
        assert_eq!(
            VecPcwFn::try_from_iters(vec![NotNan::new(1.0).unwrap()], vec![f, g]).unwrap(),
            pointwise_minimum_of_affines(x.clone().into_iter()).unwrap(),
        )
    }

    #[test]
    fn correct_double_intersect() {
        let f = AffineFunction::new(NotNan::new(1.0).unwrap(), NotNan::new(0.0).unwrap());
        let g = AffineFunction::new(NotNan::new(0.0).unwrap(), NotNan::new(1.0).unwrap());
        let h = AffineFunction::new(NotNan::new(-1.0).unwrap(), NotNan::new(3.0).unwrap());
        let mut x = [f, g, h];
        sort_affines_for_minimization(&mut x);
        assert_eq!(
            VecPcwFn::try_from_iters(
                vec![NotNan::new(1.0).unwrap(), NotNan::new(2.0).unwrap()],
                vec![f, g, h]
            )
            .unwrap(),
            pointwise_minimum_of_affines(x.clone().into_iter()).unwrap(),
        )
    }

    #[test]
    fn correct_min() {
        let f = AffineFunction::new(NotNan::new(1.0).unwrap(), NotNan::new(0.0).unwrap());
        let g = AffineFunction::new(NotNan::new(0.0).unwrap(), NotNan::new(1.0).unwrap());
        let h = AffineFunction::new(NotNan::new(-1.0).unwrap(), NotNan::new(3.0).unwrap());
        let i = AffineFunction::new(NotNan::new(0.5).unwrap(), NotNan::new(100.0).unwrap());
        let mut x = [f, g, h, i];
        sort_affines_for_minimization(&mut x);
        // dbg!(&x);
        assert_eq!(
            VecPcwFn::try_from_iters(
                vec![NotNan::new(1.0).unwrap(), NotNan::new(2.0).unwrap()],
                vec![f, g, h]
            )
            .unwrap(),
            pointwise_minimum_of_affines(x.clone().into_iter()).unwrap(),
        )
    }

    #[test]
    fn same_intercept() {
        let f = AffineFunction::new(NotNan::new(50.0).unwrap(), NotNan::new(0.0).unwrap());
        let g = AffineFunction::new(NotNan::new(1.0).unwrap(), NotNan::new(0.0).unwrap());
        let h = AffineFunction::new(NotNan::new(3.0).unwrap(), NotNan::new(0.0).unwrap());
        let i = AffineFunction::new(NotNan::new(-100.0).unwrap(), NotNan::new(0.0).unwrap());
        let mut x = [f, g, h, i];
        sort_affines_for_minimization(&mut x);
        // dbg!(&x);
        assert_eq!(
            VecPcwFn::try_from_iters(vec![NotNan::new(0.0).unwrap()], vec![f, i]).unwrap(),
            pointwise_minimum_of_affines(x.clone().into_iter()).unwrap(),
        )
    }
}
