import matplotlib.pyplot as plt
import numpy as np
import pcw_regrs_py as pr

times = np.linspace(0, 10)
values = np.hstack([5*times[:10]**2, times[10:30] - 10, (-times[30:] **
                    4 + times[30:]**3 + 10)/np.amax(-times[30:]**4 + times[30:]**3 + 10)])
ts = pr.TimeSeriesSample(times, values)

# fit with one standard error rule
p1 = pr.PcwPolynomial.fit_ose(ts)
# fit global minimizer of cv score
p2 = pr.PcwPolynomial.fit_n_best(ts)[0]
# fit with X standard error rule for X = 2
p3 = pr.PcwPolynomial.fit_xse(ts, xse_factor=2.0)
# fit with fixed penalty γ = 0.1
p4 = pr.PcwPolynomial.fit_with_penalty(ts, 0.1)


# using the native interface:
pr_native = pr._rs

# set user parameters
max_total_dof = 200
max_segment_degree = 10

# compute the full solution
sol = pr_native.fit_pcw_poly(
    ts.sample_times,
    ts.values,
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
m_as_pcw_poly = pr.PcwPolynomial.from_data_and_model(ts, m)
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
