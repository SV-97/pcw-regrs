//! Implements the solver for the dynamic program of piecewise models where the model-fitting
//! functions are indexed by their "degrees of freedom" (dof) and the subsequent CV calcuations.
mod cv;
mod opt;

pub use cv::*;
pub use opt::*;
