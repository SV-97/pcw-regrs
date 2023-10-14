import matplotlib.pyplot as plt
import numpy as np
import pcw_regrs_py as pr
import pycw_fn

times = np.linspace(0, 10)
values = np.hstack([5*times[:10]**2, times[10:30] - 10, (-times[30:] **
                    4 + times[30:]**3 + 10)/np.amax(-times[30:]**4 + times[30:]**3 + 10)])
sample = pr.TimeSeriesSample(times, values)

# fit with one standard error rule
p1 = pr.PcwPolynomial.fit_ose(sample)
# fit global minimizer of cv score
p2 = pr.PcwPolynomial.fit_n_best(sample)[0]
# fit with X standard error rule for X = 2
p3 = pr.PcwPolynomial.fit_xse(sample, xse_factor=2.0)
# fit with fixed penalty γ = 0.1
p4 = pr.PcwPolynomial.fit_with_penalty(sample, 0.1)


# using the native interface:
pr_native = pr._rs

# set user parameters
max_total_dof = 200
max_segment_degree = 10

# compute the full solution using the native interface
sol = pr_native.fit_pcw_poly(
    sample.sample_times,
    sample.values,
    max_total_dof,
    max_segment_degree + 1 if max_segment_degree is not None else None,
    weights=None,
)

# function mapping penalties to corresponding solutions of penalized partition problem
model_func = sol.model_func()
print(model_func.jump_points)
models = model_func.values
# the model corresponding to the leftmost gamma segment
m = models[0]
print(m)
# can be easily converted into a piecewise polynomial python object
m_as_pcw_poly = pr.PcwPolynomial.from_data_and_model(sample, m)
# or processed directly:
print(m.cv_score)  # score of model
print(m.cut_idxs)  # "changepoints" as indices into the timeseries
print(m.model_params)  # list of model parameters

ts = np.linspace(0, 10, 3000)
plt.scatter(times, values)
plt.scatter(ts, p2(ts), label="Best", marker=".", alpha=0.3)
plt.scatter(ts, p1(ts), label="OSE", marker=".")
plt.legend()
plt.show()

# compute the full solution using the python interface
sol = pr.Solution.fit(
    sample,
    max_total_dof,
    max_segment_degree + 1 if max_segment_degree is not None else None,
    weights=None
)

# plot the CV function
cv_func = sol.cv_func
dcv_func = sol.downsampled_cv_func
# generate sample locations for the CV function for plotting
gammas = np.logspace(np.log10(min(cv_func.jump_points[0], dcv_func.jump_points[0])) - 1, np.log10(
    max(cv_func.jump_points[-1], dcv_func.jump_points[-1])) + 1, num=100_000)

# draw both functions
plt.scatter(gammas, cv_func(gammas), marker=".", label="CV function")
plt.scatter(gammas, sol.downsampled_cv_func(gammas),
            marker=".", label="Downsampled CV function")
# add vertical lines at all points where the solution changes
plt.vlines(sol.downsampled_cv_func.jump_points, ymin=np.amin(
    cv_func.values), ymax=np.amax(cv_func.values), alpha=0.1)
# determine minimal cv score and corresponding standard error
idx_min = np.argmin(dcv_func.values)
cv_min = dcv_func.values[idx_min]
se_min = sol.downsampled_cv_se_func.values[idx_min]
# draw vertical strip showing standard error
plt.fill_between([gammas[0], gammas[-1]], [cv_min - se_min]
                 * 2, [cv_min + se_min] * 2, color="green", alpha=0.1)

plt.yscale("log", base=10)
plt.xscale("log", base=10)
plt.legend()
plt.show()
