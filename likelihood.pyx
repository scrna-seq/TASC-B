from libc.math cimport lgamma
from libc.math cimport exp
from libc.math cimport pow
from libc.math cimport log
from libc.math cimport sqrt
from scipy.optimize import minimize_scalar
from libc.math cimport pi
from scipy.integrate import quad
import cython

@cython.cdivision(True)
cdef double expit(double p):
    return 1.0 / (1 + exp(-p))

@cython.cdivision(True)
cdef double second_order_derivative_nob(double alpha, double beta, double kappa, double tau, double theta_g,
                                        double sigma_g, double mu_cg, long y_cg):
    cdef double cg

    if pi * sigma_g * sigma_g == 0:
        return float('inf')

    if y_cg == 0:
        cg = ((0.2e1 * tau * tau * pow(exp(mu_cg * tau + kappa), 0.2e1) * pow(0.1e1 + exp(mu_cg * tau + kappa), -0.3e1) - tau * tau * exp(mu_cg * tau + kappa) * pow(0.1e1 + exp(mu_cg * tau + kappa), -0.2e1) + 0.2e1 * exp(-exp(beta * mu_cg + alpha)) * tau * tau * pow(exp(-mu_cg * tau - kappa), 0.2e1) * pow(0.1e1 + exp(-mu_cg * tau - kappa), -0.3e1) - 0.2e1 * beta * exp(beta * mu_cg + alpha) * exp(-exp(beta * mu_cg + alpha)) * tau * exp(-mu_cg * tau - kappa) * pow(0.1e1 + exp(-mu_cg * tau - kappa), -0.2e1) - exp(-exp(beta * mu_cg + alpha)) * tau * tau * exp(-mu_cg * tau - kappa) * pow(0.1e1 + exp(-mu_cg * tau - kappa), -0.2e1) - beta * beta * exp(beta * mu_cg + alpha) * exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa)) + beta * beta * pow(exp(beta * mu_cg + alpha), 0.2e1) * exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) * sqrt(0.2e1) * exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) * pow(pi * sigma_g * sigma_g, -0.1e1 / 0.2e1) / 0.2e1 - 0.2e1 * (-tau * exp(mu_cg * tau + kappa) * pow(0.1e1 + exp(mu_cg * tau + kappa), -0.2e1) + exp(-exp(beta * mu_cg + alpha)) * tau * exp(-mu_cg * tau - kappa) * pow(0.1e1 + exp(-mu_cg * tau - kappa), -0.2e1) - beta * exp(beta * mu_cg + alpha) * exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) * sqrt(0.2e1) * (mu_cg - theta_g) * exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) * pow(pi * sigma_g * sigma_g, -0.1e1 / 0.2e1) * pow(sigma_g, -0.2e1) - (0.1e1 / (0.1e1 + exp(mu_cg * tau + kappa)) + exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) * sqrt(0.2e1) * exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) * pow(pi * sigma_g * sigma_g, -0.1e1 / 0.2e1) * pow(sigma_g, -0.2e1) + 0.2e1 * (0.1e1 / (0.1e1 + exp(mu_cg * tau + kappa)) + exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) * sqrt(0.2e1) * pow(mu_cg - theta_g, 0.2e1) * exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) * pow(pi * sigma_g * sigma_g, -0.1e1 / 0.2e1) * pow(sigma_g, -0.4e1)) * sqrt(0.2e1) * sqrt(pi * sigma_g * sigma_g) / (0.1e1 / (0.1e1 + exp(mu_cg * tau + kappa)) + exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) / exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) - ((-tau * exp(mu_cg * tau + kappa) * pow(0.1e1 + exp(mu_cg * tau + kappa), -0.2e1) + exp(-exp(beta * mu_cg + alpha)) * tau * exp(-mu_cg * tau - kappa) * pow(0.1e1 + exp(-mu_cg * tau - kappa), -0.2e1) - beta * exp(beta * mu_cg + alpha) * exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) * sqrt(0.2e1) * exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) * pow(pi * sigma_g * sigma_g, -0.1e1 / 0.2e1) / 0.2e1 - (0.1e1 / (0.1e1 + exp(mu_cg * tau + kappa)) + exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) * sqrt(0.2e1) * (mu_cg - theta_g) * exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) * pow(pi * sigma_g * sigma_g, -0.1e1 / 0.2e1) * pow(sigma_g, -0.2e1)) * sqrt(0.2e1) * sqrt(pi * sigma_g * sigma_g) * (-tau * exp(mu_cg * tau + kappa) * pow(0.1e1 + exp(mu_cg * tau + kappa), -0.2e1) + exp(-exp(beta * mu_cg + alpha)) * tau * exp(-mu_cg * tau - kappa) * pow(0.1e1 + exp(-mu_cg * tau - kappa), -0.2e1) - beta * exp(beta * mu_cg + alpha) * exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) * pow(0.1e1 / (0.1e1 + exp(mu_cg * tau + kappa)) + exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa)), -0.2e1) / exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) + 0.2e1 * ((-tau * exp(mu_cg * tau + kappa) * pow(0.1e1 + exp(mu_cg * tau + kappa), -0.2e1) + exp(-exp(beta * mu_cg + alpha)) * tau * exp(-mu_cg * tau - kappa) * pow(0.1e1 + exp(-mu_cg * tau - kappa), -0.2e1) - beta * exp(beta * mu_cg + alpha) * exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) * sqrt(0.2e1) * exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) * pow(pi * sigma_g * sigma_g, -0.1e1 / 0.2e1) / 0.2e1 - (0.1e1 / (0.1e1 + exp(mu_cg * tau + kappa)) + exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) * sqrt(0.2e1) * (mu_cg - theta_g) * exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) * pow(pi * sigma_g * sigma_g, -0.1e1 / 0.2e1) * pow(sigma_g, -0.2e1)) * sqrt(0.2e1) * sqrt(pi * sigma_g * sigma_g) * (mu_cg - theta_g) / (0.1e1 / (0.1e1 + exp(mu_cg * tau + kappa)) + exp(-exp(beta * mu_cg + alpha)) / (0.1e1 + exp(-mu_cg * tau - kappa))) / exp(-pow(mu_cg - theta_g, 0.2e1) * pow(sigma_g, -0.2e1)) * pow(sigma_g, -0.2e1)
    else:
        cg = -(exp((beta * mu_cg - 2 * mu_cg * tau + alpha - 2 * kappa)) * (beta * beta) * sigma_g * sigma_g + 0.2e1 * (beta * beta) * exp((beta * mu_cg - mu_cg * tau + alpha - kappa)) * sigma_g * sigma_g + (tau * tau) * exp((-mu_cg * tau - kappa)) * sigma_g * sigma_g + (beta * beta) * exp((beta * mu_cg + alpha)) * sigma_g * sigma_g + 0.2e1 * exp((-2 * mu_cg * tau - 2 * kappa)) + 0.4e1 * exp((-mu_cg * tau - kappa)) + 0.2e1) * pow(sigma_g, -0.2e1) * pow(0.1e1 + exp((-mu_cg * tau - kappa)), -0.2e1)
    return cg

cdef double log_dpois0(double log_mean):
    return -exp(log_mean)

cdef double log_dpois(long count, double log_mean):
    return count * log_mean - lgamma(count + 1) - exp(log_mean)

cdef double log_expit(double x):
    return -log(1.0 + exp(-x))

cdef double log_sum_exp2(double a, double b):
    cdef double max_el = max(a, b)
    return max_el + log(exp(a - max_el) + exp(b - max_el))

@cython.cdivision(True)
cdef double log_dnorm(double x, double mu, double sigma):
    if sigma == 0.0:
        if x == mu:
            return 0.0
        else:
            return -float('inf')
    else:
        return -0.918938533204672669540968854562379419803619384765625 - log(sigma) - (x - mu) * (
        x - mu) / sigma / sigma / 2.0

cdef double neg_log_single_complete_likelihood_nob(double mu_cg, double a_c, double b_c, double k_c, double t_c,
                                                   double theta_g, double sigma_g, long y_cg):
    if y_cg == 0:
        return -(log_sum_exp2(log_expit(-(k_c + t_c * mu_cg)),
                              log_expit(k_c + t_c * mu_cg) + log_dpois0(a_c + b_c * mu_cg)) + log_dnorm(mu_cg, theta_g,
                                                                                                        sigma_g))
    else:
        return -(log_expit(k_c + t_c * mu_cg) + log_dpois(y_cg, a_c + b_c * mu_cg) + log_dnorm(mu_cg, theta_g, sigma_g))

cdef double single_complete_likelihood_nob(double mu_cg, double a_c, double b_c, double k_c, double t_c, double theta_g,
                                           double sigma_g, long y_cg, double scale_factor_cg):
    return exp(
        -neg_log_single_complete_likelihood_nob(mu_cg, a_c, b_c, k_c, t_c, theta_g, sigma_g, y_cg) + scale_factor_cg)

cdef double neg_log_single_marginal_likelihood_nob(double a_c, double b_c, double k_c, double t_c, double theta_g,
                                                   double sigma_g, long y_cg):
    # first get the min of the neg log-likelihood
    # use brent method
    min_neg_log = minimize_scalar(neg_log_single_complete_likelihood_nob,
                                  args=(a_c, b_c, k_c, t_c, theta_g, sigma_g, y_cg), method='brent')
    cdef double min_val
    cdef double hessian
    cdef double lower_b
    cdef double upper_b
    cdef double arg_min
    if min_neg_log.success:
        arg_min = min_neg_log.x
        min_val = min_neg_log.fun
        hessian = second_order_derivative_nob(a_c, b_c, k_c, t_c, theta_g, sigma_g, arg_min, y_cg)
        lower_b = arg_min - 20 / sqrt(abs(hessian))
        upper_b = arg_min + 20 / sqrt(abs(hessian))
        integral = quad(single_complete_likelihood_nob, lower_b, upper_b,
                        args=(a_c, b_c, k_c, t_c, theta_g, sigma_g, y_cg, min_val))
        return -(log(integral[0]) - min_val)
    else:
        return float('nan')

def neg_log_sum_marginal_likelihood_nob(real_params_g, abkt, y_g):
    cdef double sum_marginal_likelihood = 0
    cdef double a_c
    cdef double b_c
    cdef double k_c
    cdef double t_c
    cdef double theta_g = real_params_g[0]
    cdef double sigma_g = exp(real_params_g[1])
    for i in range(len(y_g)):
        a_c = abkt[i, 0]
        b_c = abkt[i, 1]
        k_c = abkt[i, 2]
        t_c = abkt[i, 3]
        sum_marginal_likelihood += neg_log_single_marginal_likelihood_nob(a_c, b_c, k_c, t_c, theta_g, sigma_g, y_g[i])
    return sum_marginal_likelihood

cdef extern from "math.h":
    bint isnan(double x)

def neg_log_sum_marginal_likelihood(real_params_g, abkt, y_g):
    cdef double sum_marginal_likelihood = 0
    cdef double a_c
    cdef double b_c
    cdef double k_c
    cdef double t_c
    cdef double theta_g = real_params_g[0]
    cdef double sigma_g = exp(real_params_g[1])
    cdef double p_g = expit(real_params_g[2])
    cdef double t2 = 0
    for i in range(len(y_g)):
        a_c = abkt[i, 0]
        b_c = abkt[i, 1]
        k_c = abkt[i, 2]
        t_c = abkt[i, 3]
        t2 = neg_log_single_marginal_likelihood_nob(a_c,b_c,k_c,t_c,theta_g,sigma_g,y_g[i])
        if isnan(t2):
            return float('nan')
        if y_g[i]>0:
            sum_marginal_likelihood += (t2-log(p_g))
        elif y_g[i]==0:
            sum_marginal_likelihood += (-log_sum_exp2(log(1-p_g),-t2+log(p_g)))
    return sum_marginal_likelihood

def neg_log_sum_marginal_likelihood_free_p(real_params_g, abkt, y_g, x_g):
    cdef double sum_marginal_likelihood = 0
    cdef double a_c
    cdef double b_c
    cdef double k_c
    cdef double t_c
    cdef double theta_g
    cdef double sigma_g
    cdef double p_g
    cdef double t2 = 0
    for i in range(len(y_g)):
        a_c = abkt[i, 0]
        b_c = abkt[i, 1]
        k_c = abkt[i, 2]
        t_c = abkt[i, 3]
        theta_g = real_params_g[0]
        sigma_g = exp(real_params_g[1])
        p_g = expit(real_params_g[2]) * (1 - x_g[i]) + expit(real_params_g[3]) * x_g[i]
        t2 = neg_log_single_marginal_likelihood_nob(a_c,b_c,k_c,t_c,theta_g,sigma_g,y_g[i])
        if isnan(t2):
            return float('nan')
        if y_g[i]>0:
            sum_marginal_likelihood += (t2-log(p_g))
        elif y_g[i]==0:
            sum_marginal_likelihood += (-log_sum_exp2(log(1-p_g),-t2+log(p_g)))
    return sum_marginal_likelihood

def neg_log_sum_marginal_likelihood_free_theta(real_params_g, abkt, y_g, x_g):
    cdef double sum_marginal_likelihood = 0
    cdef double a_c
    cdef double b_c
    cdef double k_c
    cdef double t_c
    cdef double theta_g
    cdef double sigma_g
    cdef double p_g
    cdef double t2 = 0
    for i in range(len(y_g)):
        a_c = abkt[i, 0]
        b_c = abkt[i, 1]
        k_c = abkt[i, 2]
        t_c = abkt[i, 3]
        theta_g = real_params_g[0] * (1 - x_g[i]) + real_params_g[1] * x_g[i]
        sigma_g = exp(real_params_g[2])
        p_g = expit(real_params_g[3])
        t2 = neg_log_single_marginal_likelihood_nob(a_c,b_c,k_c,t_c,theta_g,sigma_g,y_g[i])
        if isnan(t2):
            return float('nan')
        if y_g[i]>0:
            sum_marginal_likelihood += (t2-log(p_g))
        elif y_g[i]==0:
            sum_marginal_likelihood += (-log_sum_exp2(log(1-p_g),-t2+log(p_g)))
    return sum_marginal_likelihood

def neg_log_sum_marginal_likelihood_free_both(real_params_g, abkt, y_g, x_g):
    cdef double sum_marginal_likelihood = 0
    cdef double a_c
    cdef double b_c
    cdef double k_c
    cdef double t_c
    cdef double theta_g
    cdef double sigma_g
    cdef double p_g
    cdef double t2 = 0
    for i in range(len(y_g)):
        a_c = abkt[i, 0]
        b_c = abkt[i, 1]
        k_c = abkt[i, 2]
        t_c = abkt[i, 3]
        theta_g = real_params_g[0] * (1 - x_g[i]) + real_params_g[1] * x_g[i]
        sigma_g = exp(real_params_g[2])
        p_g = expit(real_params_g[3]) * (1 - x_g[i]) + expit(real_params_g[4]) * x_g[i]
        t2 = neg_log_single_marginal_likelihood_nob(a_c,b_c,k_c,t_c,theta_g,sigma_g,y_g[i])
        if isnan(t2):
            return float('nan')
        if y_g[i]>0:
            sum_marginal_likelihood += (t2-log(p_g))
        elif y_g[i]==0:
            sum_marginal_likelihood += (-log_sum_exp2(log(1-p_g),-t2+log(p_g)))
    return sum_marginal_likelihood