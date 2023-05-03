# Tracking issues:

* `piecewise::PcwPolynomial` interface could be improved by changing `ASSUMED_BOUNDARY_COUNT` to `ASSUMED_SEGMENT_COUNT`. This requires [`generic_const_expr` #76560](https://doc.rust-lang.org/stable/unstable-book/language-features/generic-const-exprs.html).
* The `Add` instace of `PcwFn` may be improved using `std::cmp::max` on the output smallvec param once [`const_cmp` #92391](https://github.com/rust-lang/rust/issues/92391) is stabilized.
