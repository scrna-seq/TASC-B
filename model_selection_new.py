#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import argparse
from time import strftime
from numpy import log
from scipy.optimize import minimize
from scipy.special import expit
from scipy.special import logit
from numpy import exp
from numpy import column_stack
from numpy.random import uniform
from scipy.stats import chi2
import likelihood


class logger():
    def __init__(self, fh):
        self.mpi_fh = fh

    def log(self, string):
        self.mpi_fh.Write_shared('[' + strftime("%m/%d/%Y %H:%M:%S") + ']\t' + string + '\n')
        self.mpi_fh.Sync()

    def close(self):
        self.mpi_fh.Close()


class result_writer():
    def __init__(self, fh):
        self.mpi_fh = fh

    def write_header(self, type_comp):
        if type_comp == 1:
            self.mpi_fh.Write_shared('\t'.join(['gene', 'optim_scs', 'theta_nob', 'sigma_nob', 'theta', 'sigma', 'pg', 'lrt_stat', 'lrt_pval']) + "\n")
        elif type_comp == 2:
            self.mpi_fh.Write_shared('\t'.join(['gene', 'optim_scs', 'theta_rd_0', 'theta_rd_1', 'sigma_rd', 'pg_rd', 'theta_full_0', 'theta_full_1', 'sigma_full', 'pg_full_0', 'pg_full_1', 'lrt_stat', 'lrt_pval']) + "\n")
        elif type_comp == 3:
            self.mpi_fh.Write_shared('\t'.join(['gene', 'optim_scs', 'theta_rd', 'sigma_rd', 'pg_rd_0', 'pg_rd_1', 'theta_full_0', 'theta_full_1', 'sigma_full', 'pg_full_0', 'pg_full_1', 'lrt_stat', 'lrt_pval']) + "\n")
        elif type_comp == 4:
            self.mpi_fh.Write_shared('\t'.join(['gene', 'optim_scs', 'theta_rd', 'sigma_rd', 'pg_rd', 'theta_full_0', 'theta_full_1', 'sigma_full',  'pg_full_0', 'pg_full_1', 'lrt_stat', 'lrt_pval']) + "\n")

    def get_num_fields(self, type_comp):
        if type_comp == 1:
            return 7
        elif type_comp == 2:
            return 11
        elif type_comp == 3:
            return 11
        elif type_comp == 4:
            return 10

    def log(self, res):
        self.mpi_fh.Write_shared('\t'.join([str(x) for x in res]) + '\n')
        self.mpi_fh.Sync()

    def close(self):
        self.mpi_fh.Close()


def get_non_zero(y):
    num_non_zero=0
    for el in y:
        if el > 0:
            num_non_zero += 1
    return num_non_zero


def get_rr_range(y):
    abkt_mean = np.mean(abkt_params, axis=0)
    alpha = abkt_mean[0]
    beta = abkt_mean[1]
    theta_upper = (np.mean(np.log(y[y > 0])) - alpha) / beta
    theta_lower = (-1 - alpha) / beta
    p_upper = 0.9
    p_lower = float(np.sum(y>0)) / len(y)
    std_upper = np.std(np.log(y + 1))/beta/beta
    std_lower = np.std(np.log(y[y > 0] + 1))/beta/beta
    # std_upper = 10
    # std_lower = 1
    return theta_lower, theta_upper, p_lower, p_upper, std_lower, std_upper


def get_rr_range_grp(y, x):
    abkt_mean0 = np.mean(abkt_params[x==0,:], axis=0)
    abkt_mean1 = np.mean(abkt_params[x==1,:], axis=0)
    y_grp0 = y[x==0]
    y_grp1 = y[x==1]
    theta_upper0 = (np.mean(np.log(y_grp0[y_grp0 > 0])) - abkt_mean0[0])/abkt_mean0[1]
    theta_lower0 = (-1 - abkt_mean0[0])/abkt_mean0[1]
    theta_upper1 = (np.mean(np.log(y_grp1[y_grp1 > 0])) - abkt_mean1[0])/abkt_mean1[1]
    theta_lower1 = (-1 - abkt_mean1[0])/abkt_mean1[1]
    p_upper0 = 0.9
    p_upper1 = 0.9
    p_lower0 = float(np.sum(y_grp0 > 0)) / len(y_grp0)
    p_lower1 = float(np.sum(y_grp1 > 0)) / len(y_grp1)
    std_upper0 = np.std(np.log(y_grp0 + 1))/abkt_mean0[1]/abkt_mean0[1]
    std_lower0 = np.std(np.log(y_grp0[y_grp0 > 0] + 1))/abkt_mean0[1]/abkt_mean0[1]
    std_upper1 = np.std(np.log(y_grp1 + 1))/abkt_mean1[1]/abkt_mean1[1]
    std_lower1 = np.std(np.log(y_grp1[y_grp1 > 0] + 1))/abkt_mean1[1]/abkt_mean1[1]
    # std_upper0 = std_upper1 = 10
    # std_lower0 = std_lower1 = 1
    return theta_lower0, theta_upper0, p_lower0, p_upper0, std_lower0, std_upper0, theta_lower1, theta_upper1, p_lower1, p_upper1, std_lower1, std_upper1



def get_parsed_options():
    parser=argparse.ArgumentParser(description='TASC-B, a quantifier for gene expression incorporating gene bursting.')
    parser.add_argument('-y', '--counts', type=str, dest='y_filename', action='store', default='y.tsv',
                        help='name of the file containing the counts')
    parser.add_argument('-x', '--group', type=str, dest='x_filename', action='store', default='x.tsv',
                        help='name of the file containing group info')
    parser.add_argument('-k', '--abkt', type=str, dest='abkt_filename', action='store', default='abkt.tsv',
                        help='name of the file containing given abkt values')
    parser.add_argument('-t', '--type', type=int, dest='type_op', action='store', default=1,
                        help='type of operation: \n1 - test p < 1; \n2 - test p1 != p2, \n3 - test t1 != t2, \n4 - test 2 and 3 simultaneously')
    parser.add_argument('-o', '--outdest', type=str, dest='out_filename', action='store', default='tasc_out.tsv',
                        help='name of the output file')
    parser.add_argument('-r', '--minrestart', type=int, dest='minNR', action='store', default=2,
                        help='minimum number of restarts for optimization (default=2)')
    parser.add_argument('-m', '--maxrestart', type=int, dest='maxNR', action='store', default=8,
                        help='max number of restarts for optimization (default=8)')
    args=parser.parse_args()
    return args


def parse_filter_counts(y_filename, size):
    genes=[[] for _ in range(size)]
    log_fh.log('parsing counts file: ' + y_filename)

    with open(y_filename) as f:
        idx=0
        total_num_genes = 0
        for line in f:
            tokens=line.rstrip('\n').split('\t')
            counts=np.array([long(x) for x in tokens[1].split(',')])
            if get_non_zero(counts) >= 3:
                genes[idx].append((tokens[0], counts))
                idx += 1
                total_num_genes += 1
                if idx >= size:
                    idx=0
    log_fh.log('total number of genes parsed: ' + str(total_num_genes))
    return genes


def model0(gene_name, abkt, y_g, num_random_restarts, minrr):
    '''
    optimization without pg
    :param abkt:
    :param y_g:
    :param num_random_restarts:
    :param minrr:
    :return: min object with lowest negative log-likelihood
    '''
    theta_lower, theta_upper, p_lower, p_upper, std_lower, std_upper = get_rr_range(y_g)

    real_params_g_rtimes = column_stack((uniform(theta_lower, theta_upper, num_random_restarts),
                                         log(uniform(std_lower, std_upper, num_random_restarts))))

    arg_min_x=[]
    val_min_x=[]
    for i in range(num_random_restarts):
        log_fh.log('tasc optimization #' + str(i) + ' for gene ' + gene_name)
        real_params_g=real_params_g_rtimes[i,:]
        optim_result_obj=minimize(likelihood.neg_log_sum_marginal_likelihood_nob, x0=real_params_g, args=(abkt, y_g), method='L-BFGS-B')
        if optim_result_obj.success and (not np.isnan(optim_result_obj.fun)) and (not optim_result_obj.fun == 0):
            arg_min_x.append(optim_result_obj)
            val_min_x.append(optim_result_obj.fun)
        if len(arg_min_x) >= minrr:
            break

    if len(arg_min_x) == 0:
            return None
    else:
        return arg_min_x[np.argmin(val_min_x)]


def model1(gene_name, abkt, y_g, num_random_restarts, minrr):
    '''
    optimization with 1 pg
    :param abkt:
    :param y_g:
    :param num_random_restarts:
    :param minrr:
    :return: min object with lowest negative log-likelihood
    '''
    theta_lower, theta_upper, p_lower, p_upper, std_lower, std_upper = get_rr_range(y_g)

    real_params_g_rtimes = column_stack((uniform(theta_lower, theta_upper, num_random_restarts),
                                         log(uniform(std_lower, std_upper, num_random_restarts)),
                                         logit(uniform(p_lower, p_upper, num_random_restarts))))
    arg_min_x=[]
    val_min_x=[]

    for i in range(num_random_restarts):
        log_fh.log('tasc-b optimization #' + str(i) + ' for gene ' + gene_name)
        real_params_g=real_params_g_rtimes[i,:]
        optim_result_obj=minimize(likelihood.neg_log_sum_marginal_likelihood, x0=real_params_g, args=(abkt, y_g), method='L-BFGS-B')
        if optim_result_obj.success and (not np.isnan(optim_result_obj.fun)) and (not optim_result_obj.fun == 0):
            arg_min_x.append(optim_result_obj)
            val_min_x.append(optim_result_obj.fun)
        if len(arg_min_x) >= minrr:
            break

    if len(arg_min_x) == 0:
            return None
    else:
        return arg_min_x[np.argmin(val_min_x)]


def model2(gene_name, abkt, y_g, num_random_restarts, minrr):
    '''
    optimization with 2 pg, 1 theta
    :param abkt:
    :param y_g:
    :param num_random_restarts:
    :param minrr:
    :return: min object with lowest negative log-likelihood
    '''

    theta_lower0, theta_upper0, p_lower0, p_upper0, std_lower0, std_upper0, theta_lower1, theta_upper1, p_lower1, p_upper1, std_lower1, std_upper1 = get_rr_range_grp(
        y_g, group_info)
    real_params_g_rtimes = column_stack(
        (uniform(min(theta_lower0, theta_lower1), max(theta_upper0, theta_upper1), num_random_restarts),
         log(uniform(min(std_lower0, std_lower1), max(std_upper0, std_upper1), num_random_restarts)),
         logit(uniform(p_lower0, p_upper0, num_random_restarts)),
         logit(uniform(p_lower1, p_upper1, num_random_restarts))))
    arg_min_x = []
    val_min_x = []

    for i in range(num_random_restarts):
        log_fh.log('tasc free p optimization #' + str(i) + ' for gene ' + gene_name)
        real_params_g = real_params_g_rtimes[i, :]
        optim_result_obj = minimize(likelihood.neg_log_sum_marginal_likelihood_free_p, x0=real_params_g,
                                    args=(abkt, y_g, group_info), method='L-BFGS-B')
        if optim_result_obj.success and (not np.isnan(optim_result_obj.fun)) and (optim_result_obj.fun != 0):
            arg_min_x.append(optim_result_obj)
            val_min_x.append(optim_result_obj.fun)
        if len(arg_min_x) >= minrr:
            break

    if len(arg_min_x) == 0:
            return None
    else:
        return arg_min_x[np.argmin(val_min_x)]


def model3(gene_name, abkt, y_g, num_random_restarts, minrr):
    '''
    optimization with 1 pg, 2 theta
    :param abkt:
    :param y_g:
    :param num_random_restarts:
    :param minrr:
    :return: min object with lowest negative log-likelihood
    '''
    theta_lower0, theta_upper0, p_lower0, p_upper0, std_lower0, std_upper0, theta_lower1, theta_upper1, p_lower1, p_upper1, std_lower1, std_upper1 = get_rr_range_grp(
        y_g, group_info)
    real_params_g_rtimes = column_stack((uniform(theta_lower0, theta_upper0, num_random_restarts),
                                         uniform(theta_lower1, theta_upper1, num_random_restarts),
                                         log(uniform(min(std_lower0, std_lower1), max(std_upper0, std_upper1), num_random_restarts)),
                                         logit(uniform(min(p_lower0, p_lower1), max(p_upper0, p_upper1), num_random_restarts))))
    arg_min_x=[]
    val_min_x=[]
    for i in range(num_random_restarts):
        log_fh.log('tasc free theta optimization #' + str(i) + ' for gene ' + gene_name)
        real_params_g=real_params_g_rtimes[i,:]
        optim_result_obj=minimize(likelihood.neg_log_sum_marginal_likelihood_free_theta, x0=real_params_g, args=(abkt, y_g, group_info), method='L-BFGS-B')
        if optim_result_obj.success and (not np.isnan(optim_result_obj.fun)) and (optim_result_obj.fun != 0):
            arg_min_x.append(optim_result_obj)
            val_min_x.append(optim_result_obj.fun)
        if len(arg_min_x) >= minrr:
            break

    if len(arg_min_x) == 0:
            return None
    else:
        return arg_min_x[np.argmin(val_min_x)]

def model4(gene_name, abkt, y_g, num_random_restarts, minrr):
    '''
    optimization with 2 pg, 2 theta
    :param abkt:
    :param y_g:
    :param num_random_restarts:
    :param minrr:
    :return: min object with lowest negative log-likelihood
    '''
    theta_lower0, theta_upper0, p_lower0, p_upper0, std_lower0, std_upper0, theta_lower1, theta_upper1, p_lower1, p_upper1, std_lower1, std_upper1 = get_rr_range_grp(
        y_g, group_info)
    real_params_g_rtimes = column_stack((uniform(theta_lower0, theta_upper0, num_random_restarts),
                                         uniform(theta_lower1, theta_upper1, num_random_restarts),
                                         log(uniform(min(std_lower0, std_lower1), max(std_upper0, std_upper1), num_random_restarts)),
                                         logit(uniform(p_lower0, p_upper0, num_random_restarts)),
                                         logit(uniform(p_lower1, p_upper1, num_random_restarts))))
    arg_min_x=[]
    val_min_x=[]
    for i in range(num_random_restarts):
        log_fh.log('tasc free both optimization #' + str(i) + ' for gene ' + gene_name)
        real_params_g=real_params_g_rtimes[i,:]
        optim_result_obj=minimize(likelihood.neg_log_sum_marginal_likelihood_free_both, x0=real_params_g, args=(abkt, y_g, group_info), method='L-BFGS-B')
        if optim_result_obj.success and (not np.isnan(optim_result_obj.fun)) and (optim_result_obj.fun != 0):
            arg_min_x.append(optim_result_obj)
            val_min_x.append(optim_result_obj.fun)
        if len(arg_min_x) >= minrr:
            break

    if len(arg_min_x) == 0:
            return None
    else:
        return arg_min_x[np.argmin(val_min_x)]


def opt_neg_log_sum_marginal_likelihood(gene_name, abkt, y_g, num_random_restarts, minrr):
    tasc_nob_res = model0(gene_name, abkt, y_g, num_random_restarts, minrr)
    if tasc_nob_res is None:
        res_fh.log((gene_name, False)+tuple([float('nan')]*res_fh.get_num_fields(1)))
        return

    tasc_b_res = model1(gene_name, abkt, y_g, num_random_restarts, minrr)
    if tasc_b_res is None:
        res_fh.log((gene_name, False) + tuple([float('nan')] * res_fh.get_num_fields(1)))
        return

    lrt_stat = 2 * (tasc_nob_res.fun - tasc_b_res.fun)
    lrt_pval = 1 - chi2.cdf(lrt_stat, df=1)
    res_fh.log((gene_name, True, tasc_nob_res.x[0], exp(tasc_nob_res.x[1]), tasc_b_res.x[0], exp(tasc_b_res.x[1]), expit(tasc_b_res.x[2]), lrt_stat, lrt_pval))


def lrt_1p_vs_2p(gene_name, abkt, y_g, num_random_restarts, minrr):
    tasc_free_both = model4(gene_name, abkt, y_g, num_random_restarts, minrr)
    if tasc_free_both is None:
        res_fh.log((gene_name, False) + tuple([float('nan')] * res_fh.get_num_fields(2)))
        return

    tasc_free_theta = model3(gene_name, abkt, y_g, num_random_restarts, minrr)
    if tasc_free_theta is None:
        res_fh.log((gene_name, False) + tuple([float('nan')] * res_fh.get_num_fields(2)))
        return

    lrt_stat = 2 * (tasc_free_theta.fun - tasc_free_both.fun)
    lrt_pval = 1 - chi2.cdf(lrt_stat, df=1)

    res_fh.log(((gene_name, True, tasc_free_theta.x[0], tasc_free_theta.x[1], exp(tasc_free_theta.x[2]), expit(tasc_free_theta.x[3]),
                 tasc_free_both.x[0], tasc_free_both.x[1],exp(tasc_free_both.x[2]), expit(tasc_free_both.x[3]),
                expit(tasc_free_both.x[4]), lrt_stat, lrt_pval)))


def lrt_1theta_vs_2theta(gene_name, abkt, y_g, num_random_restarts, minrr):
    tasc_free_both = model4(gene_name, abkt, y_g, num_random_restarts, minrr)
    if tasc_free_both is None:
        res_fh.log((gene_name, False) + tuple([float('nan')] * res_fh.get_num_fields(3)))
        return

    tasc_free_p = model2(gene_name, abkt, y_g, num_random_restarts, minrr)
    if tasc_free_p is None:
        res_fh.log((gene_name, False) + tuple([float('nan')] * res_fh.get_num_fields(3)))
        return

    lrt_stat = 2 * (tasc_free_p.fun - tasc_free_both.fun)
    lrt_pval = 1 - chi2.cdf(lrt_stat, df=1)
    res_fh.log((gene_name, True, tasc_free_p.x[0], exp(tasc_free_p.x[1]), expit(tasc_free_p.x[2]), expit(tasc_free_p.x[3]),
                tasc_free_both.x[0], tasc_free_both.x[1], exp(tasc_free_both.x[2]),
                expit(tasc_free_both.x[3]), expit(tasc_free_both.x[4]), lrt_stat, lrt_pval))


def lrt_free_p_and_theta(gene_name, abkt, y_g, num_random_restarts, minrr):
    tasc_free_both = model4(gene_name, abkt, y_g, num_random_restarts, minrr)
    if tasc_free_both is None:
        res_fh.log((gene_name, False) + tuple([float('nan')] * res_fh.get_num_fields(4)))
        return

    tasc_b_res = model1(gene_name, abkt, y_g, num_random_restarts, minrr)
    if tasc_b_res is None:
        res_fh.log((gene_name, False) + tuple([float('nan')] * res_fh.get_num_fields(4)))
        return

    lrt_stat = 2 * (tasc_b_res.fun - tasc_free_both.fun)
    lrt_pval = 1 - chi2.cdf(lrt_stat, df=2)

    res_fh.log((gene_name, True, tasc_b_res.x[0], exp(tasc_b_res.x[1]), expit(tasc_b_res.x[2]),
                tasc_free_both.x[0], tasc_free_both.x[1], exp(tasc_free_both.x[2]),
                expit(tasc_free_both.x[3]), expit(tasc_free_both.x[4]), lrt_stat, lrt_pval))


def get_min_marginal(data):
    if args.type_op == 1:
        for el in data:
            log_fh.log('now analyzing ' + el[0] + ' on node #' + str(rank))
            opt_neg_log_sum_marginal_likelihood(el[0], py_stan_input['abkt'], el[1], args.maxNR, args.minNR)
    elif args.type_op == 2:
        for el in data:
            log_fh.log('now analyzing ' + el[0] + ' on node #' + str(rank))
            lrt_1p_vs_2p(el[0], py_stan_input['abkt'], el[1], args.maxNR, args.minNR)
    elif args.type_op == 3:
        for el in data:
            log_fh.log('now analyzing ' + el[0] + ' on node #' + str(rank))
            lrt_1theta_vs_2theta(el[0], py_stan_input['abkt'], el[1], args.maxNR, args.minNR)
    elif args.type_op == 4:
        for el in data:
            log_fh.log('now analyzing ' + el[0] + ' on node #' + str(rank))
            lrt_free_p_and_theta(el[0], py_stan_input['abkt'], el[1], args.maxNR, args.minNR)


np.seterr(all='ignore')

#parse args
args=get_parsed_options()

# init mpi env
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

# init logger file handle
log_fh = logger(MPI.File.Open(comm, args.out_filename + '.log', MPI.MODE_CREATE | MPI.MODE_WRONLY))
res_fh = result_writer(MPI.File.Open(comm, args.out_filename, MPI.MODE_CREATE | MPI.MODE_WRONLY))

# all nodes init
genes_grouped_by_worker=None
abkt_params=None
py_stan_input=None
tasc_sm=None
group_info=None

# master node init
if rank == 0:
    log_fh.log('opened MPI World with size ' + str(size))
    log_fh.log('input counts filename: ' + str(args.y_filename))
    log_fh.log('input abkt filename: ' + str(args.abkt_filename))
    log_fh.log('output filename: ' + str(args.out_filename))
    log_fh.log('max number of restarts: ' + str(args.maxNR))
    log_fh.log('min number of restarts: ' + str(args.minNR))

    log_fh.log('parsing abkt file: ' + args.abkt_filename)
    res_fh.write_header(args.type_op)
    abkt_params = np.genfromtxt(args.abkt_filename)

    if (not args.type_op == 1):
        log_fh.log('parsing x file: ' + args.x_filename)
        group_info = np.genfromtxt(args.x_filename, dtype=np.int8)

    py_stan_input={
        'C': abkt_params.shape[0],
        'abkt' : abkt_params
    }

    genes_grouped_by_worker=parse_filter_counts(args.y_filename, size)

part_data = comm.scatter(genes_grouped_by_worker, root=0)
log_fh.log('rank ' + str(rank) + ' has ' + str(len(part_data)) + ' genes. the first gene is ' + part_data[0][0])

py_stan_input = comm.bcast(py_stan_input, root=0)
abkt_params = comm.bcast(abkt_params, root=0)
group_info = comm.bcast(group_info, root=0)

opt_marg_results = get_min_marginal(part_data)

log_fh.close()
res_fh.close()