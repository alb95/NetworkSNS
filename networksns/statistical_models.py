import networkx as nx
import numpy as np
from scipy.sparse import identity, diags, csr_matrix, bmat, kron
from scipy.integrate import quad
from numpy.linalg import norm
from numpy import ones, zeros, arange, inf
from math import exp
import bisect
from scipy.optimize import root_scalar
from scipy.stats import norm as normal
import warnings
from numpy.random import choice
from statsmodels.tsa.arima.model import ARIMA
import csv


def darn_simulation(p, Q, Y, n, s, z_p='uniform', param='local'):
    """
    Simulate a temporal network following the :math:`DARN(p)` model.

    For a temporal network described by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}` the :math:`DARN(p)` model [1]_ describes the state :math:`A^{t}_{ij}` of the link :math:`(i,j)`  at time :math:`t` with the following copying mechanism: (i) with probability :math:`q_{ij}`, the link is copied from the past itself, (ii) then the lag is selected with probability :math:`z_i` with :math:`i=1,\\ldots,p`, where :math:`p` represents the order of the memory, (iii) while with probability :math:`1-q_{ij}`, it is sampled with (marginal) probability :math:`y_{ij}`.

    In other words, the state :math:`A_{ij}^t` of the link :math:`(i,j)` is described by the following stochastic process at discrete time

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}A^{t}_{ij}= Q^{t}_{ij} A^{t-Z_p^{t}}_{ij} + (1-Q^{t}_{ij}) Y^{t}_{ij}`

    with :math:`Q_{ij}^t\sim B(q_{ij})` Bernoulli variable, :math:`Z_p^t` random variable picking values from :math:`1` to :math:`p` with probability :math:`z_i` :math:`i=1,\\ldots,p`, respectively, and :math:`Y_{ij}^t\\sim B(y_{ij})` Bernoulli variable.

    Parameters
    __________

    p: integer
        Markov order
    Q: array_like
        Symmetric matrix. It represents the link-specific probability of copying from the past.
        A float input is accepted in the case of homogeneous copying probabilities for each link.
    Y: array_like
    Symmetric matrix.
    Bernoulli marginal probabilities.
        A float input is accepted in the case of homogeneous marginal probabilities for each link.
    n: integer
        number of nodes in the graph.
    s: integer
        Temporal Network sample size, *i.e.* the length of the time series of adjacency matrices.
    z_p: array/string, optional
        stochastic array of length :math:`p` representing the memory distribution \
        :math:`Z_p`. Default: ``'uniform'``.
        Possible strings:

        * ``'uniform'``: :math:`Z_p = (\\frac{1}{p}, \\dots, \\frac{1}{p}`),

        * ``'exponential'``: :math:`Z_p = \\frac{(1, e^{-1}, \\dots, e^{-p+1})}{\\left|\\left|(1, e^{-1}, \\dots, e^{-p+1}\\right|\\right|_2)}`

        * ``'normal'``: let :math:`f(x)=\\frac{e^{-(x-1)^2/2}}{\\sqrt{2\\pi}}` then :math:`Z_p = \\frac{(f(0), \\dots, f(p-1))}{\\left|\\left|(f(0), \\dots, f(p-1))\\right|\\right|}`
    param:
        If ``'local'`` the value of the parameters depends on the couple :math:`(i,j)`, *i.e.* parameters are link-specific, if ``'global'`` it does not depend on the :math:`(i,j)`. Default: ``'local'``.


    :return: **simulation**: (list)
        Temporal network produced by a :math:`DARN(p)` model.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Define input parameters

    .. code:: python

     >>>    n = 50
     >>>    s = 100
     >>>    p = 3
     >>>    Q = (np.ones((n, n)) - np.diag(np.ones(n))) * 0.5
     >>>    Y = (np.ones((n, n)) - np.diag(np.ones(n)))  * 0.3

    Simulate the temporal network

     .. code:: python

    >>>   time_series = sm.darn_simulation(p, Q, Y, n, s)


    References
    ----------
    .. [1] Williams, O.E., Lillo, F. and Latora, V., 2019.
           Effects of memory on spreading processes in non-Markovian temporal networks.
           New Journal of Physics, 21(4), p.043028.
           https://iopscience.iop.org/article/10.1088/1367-2630/ab13fb/meta
    """
    simulation = []
    if isinstance(z_p, str):
        if z_p == 'uniform':
            z_p = np.ones((p, 1)) / p
        elif z_p == 'exponential':
            z_p = np.exp(-np.array(range(p))) / np.sum(np.exp(-np.array(range(p))))
        elif z_p == 'normal':
            z_p = normal.pdf(np.array(range(p)), 2, 1) / np.sum(normal.pdf(np.array(range(p)), 2, 1))
        else:
            raise ValueError('The admissible strings for z_p are ' " 'uniform' "', ' " 'exponential'" ' or ' " 'normal'"
                             ', otherwise use arrays' '.')
    if param != 'local' and param != 'global':
        raise ValueError('The value of parameters must be ' " 'local' "'  or ' " 'global'" '.')
    elif param == 'local':
        simulation = [np.zeros((n, n)) for _ in range(s)]
        for j in range(n):
            for i in range(j + 1, n):
                x = choice(np.array([0, 1]), p, p=[1 / 2, 1 / 2])
                for k in range(s):
                    z = choice(list(np.append(x[-p:], np.array([1, 0]))), 1,
                               p=list(np.append(z_p * Q[i, j],
                                                np.array([(1 - Q[i, j]) * Y[i, j], (1 - Q[i, j]) * (1 - Y[i, j])]))))
                    x = np.append(x, z)
                x = x[p:]
                for pos in range(len(x)):
                    simulation[pos][i, j] = x[pos]
                    simulation[pos][j, i] = x[pos]
    elif param == 'global':
        if isinstance(Q, float) or isinstance(Q, int):
            if isinstance(Y, float) or isinstance(Y, int):
                simulation = []
                x = choice(np.array([0, 1]), p, p=[1 / 2, 1 / 2])
                for k in range(s):
                    z = choice(list(np.append(x[-p:], np.array([1, 0]))), 1,
                               p=list(np.append(z_p * Q, np.array([(1 - Q) * Y, (1 - Q) * (1 - Y)]))))
                    x = np.append(x, z)
                x = x[p:]
                for pos in range(len(x)):
                    simulation.append(x[pos] * (np.ones((n, n)) - np.eye(n)))
        else:
            simulation = []
            x = choice(np.array([0, 1]), p, p=[1 / 2, 1 / 2])
            for k in range(s):
                z = choice(list(np.append(x[-p:], np.array([1, 0]))), 1,
                           p=list(np.append(z_p * Q[0, 1], np.array(
                               [(1 - Q[0, 1]) * Y[0, 1], (1 - Q[0, 1]) * (1 - Y[0, 1])]))))
                x = np.append(x, z)
            x = x[p:]
            for pos in range(len(x)):
                simulation.append(x[pos] * (np.ones((n, n)) - np.eye(n)))
    return simulation


def darn(time_series, p, z_p='uniform', param='local'):
    """
    Estimate, by maximum likelihood method, the parameters of the :math:`DARN(p)` model.

    For a temporal network represented by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}`, the :math:`DARN(p)` model [1]_ describes the state :math:`A^{t}_{ij}` of the link :math:`(i,j)`  at time :math:`t` with the following copying mechanism: (i) with probability :math:`q_{ij}`, the link is copied from the past itself, (ii) then the lag is selected with probability :math:`z_i` with :math:`i=1,\\ldots,p`, where :math:`p` represents the order of the memory, (iii) while with probability :math:`1-q_{ij}`, it is sampled with (marginal) probability :math:`y_{ij}`.

    In other words, the state :math:`A_{ij}^t` of the link :math:`(i,j)` is described by the following stochastic process at discrete time,

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}A^{t}_{ij}= Q^{t}_{ij} A^{t-Z_p^{t}}_{ij} + (1-Q^{t}_{ij}) Y^{t}_{ij},`

    with :math:`Q_{ij}^t\\sim B(q_{ij})` Bernoulli variable, :math:`Z_p^t` random variable picking values from :math:`1` to :math:`p` with probability :math:`z_i` :math:`i=1,\\ldots,p`, respectively, and :math:`Y_{ij}^t\\sim B(y_{ij})` Bernoulli variable.



    Parameters
    __________

    time_series: List
        Time series. List of symmetric adjacency matrices [:math:`A_0, A_1, \\dots, A_T`].
    p: integer
        Markov order.
    z_p: array or string, optional
        stochastic array of length :math:`p` representing the memory distribution \
        :math:`Z_p`. Default: ``'uniform'``.
        Possible strings:

        * ``'uniform'``: :math:`Z_p = (\\frac{1}{p}, \\dots, \\frac{1}{p}`),

        * ``'exponential'``: :math:`Z_p = \\frac{(1, e^{-1}, \\dots, e^{-p+1})}{\\left|\\left|(1, e^{-1}, \\dots, e^{-p+1}\\right|\\right|_2)}`

        * ``'normal'``: let :math:`f(x)=\\frac{e^{-(x-1)^2/2}}{\\sqrt{2\\pi}}` then :math:`Z_p = \\frac{(f(0), \\dots, f(p-1))}{\\left|\\left|(f(0), \\dots, f(p-1))\\right|\\right|}`
    param: string, optional
        If ``'local'`` the parameters are different for each link. If ``'global'`` the parameters do not \
        depend on the link.  Default: ``'local'``.

    Returns
    ________

    Q: array_like
        Symmetric matrix with the estimated probabilities of copying from the past.
    Y: array_like
        Symmetric matrix with the estimated parameters for the Bernoulli trials.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Create temporal network

    .. code:: python

     >>>    n = 50
     >>>    s = 100
     >>>    p = 3
     >>>    Q = (np.ones((n, n)) - np.diag(np.ones(n))) * 0.5
     >>>    Y = (np.ones((n, n)) - np.diag(np.ones(n)))  * 0.3
     >>>    time_series = sm.darn_simulation(p, Q, Y, n, s)

    Estimate the :math:`DARN(p)` model parameters

     .. code:: python

    >>>   Q, Y = sm.darn(time_series, p)

    References
    __________

    .. [1] Williams, O.E., Lillo, F. and Latora, V., 2019.
       Effects of memory on spreading processes in non-Markovian temporal networks.
       New Journal of Physics, 21(4), p.043028.
       https://iopscience.iop.org/article/10.1088/1367-2630/ab13fb/meta
    """
    n = time_series[0].shape[0]

    # check for constant entries by aggregating the time series.

    aggregation = csr_matrix((n, n))
    T = len(time_series)
    for i in range(T):
        aggregation = aggregation + time_series[i]
    aggregation = aggregation - np.diag(np.diag(aggregation))
    # Assume that the model uses these parameters: X_t = Q_t X_{t-Z_t} + (1-Q_t) Y_t, Q^{ij}_t bernoulli(q^{ij}),
    # Y_t = bernoulli(y^{ij})

    Q = np.zeros((n, n))
    Y = np.zeros((n, n))

    if param == 'local':
        if p == 1:
            for i in range(1, n):
                for j in range(0, i):
                    if aggregation[i, j] == 0:
                        warnings.warn(
                            'Constant entries detected: parameters cannot be estimated')
                        # set q = 0 and y = 0
                        Q[i, j] = 0
                        Q[j, i] = Q[i, j]
                        Y[i, j] = 0
                        Y[j, i] = Y[i, j]
                    elif aggregation[i, j] == T:
                        warnings.warn(
                            'Constant entries detected: parameters cannot be estimated')
                        # set q = 1 and y = 0
                        Q[i, j] = 0
                        Q[j, i] = Q[i, j]
                        Y[i, j] = 1
                        Y[j, i] = Y[i, j]
                    else:
                        bound_0 = 0.001
                        bound_1 = 0.999
                        q_0 = aggregation[i, j] / T
                        y_0 = 1 / 3
                        A = np.array([time_series[t][i, j] for t in range(T)])
                        A_minus = A[:(T - 1)]
                        A_plus = A[1:]
                        r_0 = __likelihood_d_q(bound_0, y_0, A_plus, A_minus)
                        r_1 = __likelihood_d_q(bound_1, y_0, A_plus, A_minus)
                        if r_0 * r_1 >= 0:
                            y_0 = y_0
                            q_0 = 0
                        else:
                            y_1 = 2 / 3
                            tol = 1e-8
                            maxit = 100
                            precision = 1
                            it = 0
                            while precision > tol and it < maxit:
                                q_1 = root_scalar(__likelihood_d_q, args=(y_0, A_plus, A_minus),
                                                  bracket=(bound_0, bound_1)).root
                                c_0 = __likelihood_d_y(bound_0, q_1, A_plus, A_minus)
                                c_1 = __likelihood_d_y(bound_1, q_1, A_plus, A_minus)
                                if c_0 * c_1 < 0:
                                    y_1 = root_scalar(__likelihood_d_y, args=(q_1, A_plus, A_minus),
                                                      bracket=(bound_0, bound_1)).root
                                else:
                                    q_1 = aggregation[i, j] / T
                                precision = max(abs((y_1 - y_0) / y_0), abs((q_1 - q_0) / q_0))
                                it = it + 1
                                y_0 = y_1
                                q_0 = q_1
                        Q[i, j] = q_0
                        Y[i, j] = y_0
                        Q[j, i] = Q[i, j]
                        Y[j, i] = Y[i, j]

        elif p > 1:
            if isinstance(z_p, str):
                if z_p == 'uniform':
                    z_p = np.ones((p, 1)) / p
                elif z_p == 'exponential':
                    z_p = np.exp(-np.array(range(p))) / np.sum(np.exp(-np.array(range(p))))
                elif z_p == 'normal':
                    z_p = normal.pdf(np.array(range(p)), 2, 1) / np.sum(normal.pdf(np.array(range(p)), 2, 1))
                else:
                    raise ValueError(
                        'The admissible strings for z_p are ' " 'uniform' "', ' " 'exponential'" ' or ' " 'normal'"
                        ', otherwise use arrays' '.')
            for i in range(1, n):
                for j in range(0, i):
                    if aggregation[i, j] == 0:
                        warnings.warn(
                            'Constant entries detected: parameters cannot be estimated')
                        # set q = 0 and y = 0
                        Q[i, j] = 0
                        Q[j, i] = Q[i, j]
                        Y[i, j] = 0
                        Y[j, i] = Y[i, j]
                    elif aggregation[i, j] == T:
                        warnings.warn(
                            'Constant entries detected: parameters cannot be estimated')
                        # set q = 1 and y = 0
                        Q[i, j] = 0
                        Q[j, i] = Q[i, j]
                        Y[i, j] = 1
                        Y[j, i] = Y[i, j]
                    else:
                        bound_0 = 0.001
                        bound_1 = 0.999
                        q_0 = aggregation[i, j] / T
                        y_0 = 2 / 3
                        A = np.array([time_series[t][i, j] for t in range(T)])
                        r_0 = __likelihood_p_d_q(bound_0, y_0, p, z_p, A)
                        r_1 = __likelihood_p_d_q(bound_1, y_0, p, z_p, A)
                        if r_0 * r_1 >= 0:
                            y_0 = y_0
                            q_0 = 0
                        else:
                            y_1 = 1 / 3
                            tol = 1e-8
                            maxit = 100
                            precision = 1
                            it = 0
                            while precision > tol and it < maxit:
                                q_1 = root_scalar(__likelihood_p_d_q, args=(y_0, p, z_p, A),
                                                  bracket=(bound_0, bound_1)).root
                                c_0 = __likelihood_p_d_y(bound_0, q_1, p, z_p, A)
                                c_1 = __likelihood_p_d_y(bound_1, q_1, p, z_p, A)
                                if c_0 * c_1 < 0:
                                    y_1 = root_scalar(__likelihood_p_d_y, args=(q_1, p, z_p, A),
                                                      bracket=(bound_0, bound_1)).root
                                else:
                                    q_1 = aggregation[i, j] / T
                                precision = max(abs((y_1 - y_0) / y_0), abs((q_1 - q_0) / q_0))
                                it = it + 1
                                y_0 = y_1
                                q_0 = q_1
                        Q[i, j] = q_0
                        Y[i, j] = y_0
                        Q[j, i] = Q[i, j]
                        Y[j, i] = Y[i, j]

    if param == 'global':
        if p == 1:
            bound_0 = 0.001
            bound_1 = 0.999
            q_0 = np.sum(np.sum(aggregation)) / (n * (n - 1) * T)
            y_0 = 1 / 3
            r_0 = __global_likelihood_d_q(bound_0, y_0, time_series)
            r_1 = __global_likelihood_d_q(bound_1, y_0, time_series)
            if r_0 * r_1 >= 0:
                y_0 = y_0
                q_0 = 0
            else:
                y_1 = 2 / 3
                tol = 1e-8
                maxit = 100
                precision = 1
                it = 0
                while precision > tol and it < maxit:
                    r_0 = __global_likelihood_d_q(bound_0, y_0, time_series)
                    r_1 = __global_likelihood_d_q(bound_1, y_0, time_series)
                    if r_0 * r_1 >= 0:
                        y_0 = y_0
                        q_0 = 0
                    else:
                        q_1 = root_scalar(__global_likelihood_d_q, args=(y_0, time_series),
                                          bracket=(bound_0, bound_1)).root
                        c_0 = __global_likelihood_d_y(bound_0, q_1, time_series)
                        c_1 = __global_likelihood_d_y(bound_1, q_1, time_series)
                        if c_0 * c_1 < 0:
                            y_1 = root_scalar(__global_likelihood_d_y, args=(q_1, time_series),
                                              bracket=(bound_0, bound_1)).root
                        else:
                            q_1 = np.sum(np.sum(aggregation)) / (n * (n - 1) * T)
                        precision = max(abs((y_1 - y_0) / y_0), abs((q_1 - q_0) / q_0))
                        y_0 = y_1
                        q_0 = q_1
                    it = it + 1
            Q = np.ones([n, n]) * q_0
            Y = np.ones([n, n]) * y_0
            Q = Q - np.diag(np.diag(Q))
            Y = Y - np.diag(np.diag(Y))

        elif p > 1:
            if isinstance(z_p, str):
                if z_p == 'uniform':
                    z_p = np.ones((p, 1)) / p
                elif z_p == 'exponential':
                    z_p = np.exp(-np.array(range(p))) / np.sum(np.exp(-np.array(range(p))))
                elif z_p == 'normal':
                    z_p = normal.pdf(np.array(range(p)), 2, 1) / np.sum(normal.pdf(np.array(range(p)), 2, 1))
                else:
                    raise ValueError(
                        'The admissible strings for z_p are ' " 'uniform' "', ' " 'exponential'" ' or ' " 'normal'"
                        ', otherwise use arrays' '.')
            bound_0 = 0.001
            bound_1 = 0.999
            q_0 = np.sum(np.sum(aggregation)) / (n * (n - 1) * T)
            y_0 = 1 / 3
            r_0 = __global_likelihood_p_d_q(bound_0, y_0, p, z_p, time_series)
            r_1 = __global_likelihood_p_d_q(bound_1, y_0, p, z_p, time_series)
            if r_0 * r_1 >= 0:
                y_0 = y_0
                q_0 = 0
            else:
                y_1 = 2 / 3
                tol = 1e-8
                maxit = 100
                precision = 1
                it = 0
                while precision > tol and it < maxit:
                    r_0 = __global_likelihood_p_d_q(bound_0, y_0, p, z_p, time_series)
                    r_1 = __global_likelihood_p_d_q(bound_1, y_0, p, z_p, time_series)
                    if r_0 * r_1 >= 0:
                        y_0 = y_0
                        q_0 = 0
                    else:
                        q_1 = root_scalar(__global_likelihood_p_d_q, args=(y_0, p, z_p, time_series),
                                          bracket=(bound_0, bound_1)).root
                        c_0 = __global_likelihood_p_d_y(bound_0, q_1, p, z_p, time_series)
                        c_1 = __global_likelihood_p_d_y(bound_1, q_1, p, z_p, time_series)
                        if c_0 * c_1 < 0:
                            y_1 = root_scalar(__global_likelihood_p_d_y, args=(q_1, p, z_p, time_series),
                                              bracket=(bound_0, bound_1)).root
                        else:
                            q_1 = np.sum(np.sum(aggregation)) / (n * (n - 1) * T)
                        precision = max(abs((y_1 - y_0) / y_0), abs((q_1 - q_0) / q_0))
                        y_0 = y_1
                        q_0 = q_1
                    it = it + 1
            Q = np.ones([n, n]) * q_0
            Y = np.ones([n, n]) * y_0
            Q = Q - np.diag(np.diag(Q))
            Y = Y - np.diag(np.diag(Y))

    return Q, Y


def __likelihood_d_q(x, y, A_plus, A_minus):
    d_q = np.sum(A_plus * (A_minus - y) / (x * A_minus + (1 - x) * y) - (1 - A_plus) * (A_minus - y) / (
            x * (1 - A_minus) + (1 - x) * (1 - y)))
    return d_q


def __likelihood_d_y(x, q, A_plus, A_minus):
    d_y = np.sum(A_plus / (q * A_minus + (1 - q) * x) - (1 - A_plus) / (q * (1 - A_minus) + (1 - q) * (1 - x)))
    return d_y


def __likelihood_p_d_q(x, y, p, z_p, A):
    A_minus = np.zeros(len(A) - p)
    for i in range(1, p + 1):
        A_minus = A_minus + A[p - i:len(A) - i] * z_p[i - 1]
    A_plus = A[p:]
    d_q = __likelihood_d_q(x, y, A_plus, A_minus)
    return d_q


def __likelihood_p_d_y(x, q, p, z_p, A):
    A_minus = np.zeros(len(A) - p)
    for i in range(1, p + 1):
        A_minus = A_minus + A[p - i:len(A) - i] * z_p[i - 1]
    A_plus = A[p:]
    d_y = __likelihood_d_y(x, q, A_plus, A_minus)
    return d_y


def __global_likelihood_d_q(x, y, time_series):
    d_q = 0
    T = len(time_series)
    n = time_series[0].shape[0]
    for i in range(1, n):
        for j in range(0, i):
            A = np.array([time_series[t][i, j] for t in range(T)])
            A_minus = A[:(T - 1)]
            A_plus = A[1:]
            d_q = d_q + __likelihood_d_q(x, y, A_plus, A_minus)
    return d_q


def __global_likelihood_d_y(x, q, time_series):
    d_y = 0
    T = len(time_series)
    n = time_series[0].shape[0]
    for i in range(1, n):
        for j in range(0, i):
            A = np.array([time_series[t][i, j] for t in range(T)])
            A_minus = A[:(T - 1)]
            A_plus = A[1:]
            d_y = d_y + __likelihood_d_y(x, q, A_plus, A_minus)
    return d_y


def __global_likelihood_p_d_q(x, y, p, z_p, time_series):
    d_q = 0
    T = len(time_series)
    n = time_series[0].shape[0]
    for i in range(1, n):
        for j in range(0, i):
            A = np.array([time_series[t][i, j] for t in range(T)])
            A_minus = np.zeros(len(A) - p)
            for k in range(1, p + 1):
                A_minus = A_minus + A[p - k:len(A) - k] * z_p[k - 1]
            A_plus = A[p:]
            d_q = d_q + __likelihood_d_q(x, y, A_plus, A_minus)
    return d_q


def __global_likelihood_p_d_y(x, q, p, z_p, time_series):
    d_y = 0
    T = len(time_series)
    n = time_series[0].shape[0]
    for i in range(1, n):
        for j in range(0, i):
            A = np.array([time_series[t][i, j] for t in range(T)])
            A_minus = np.zeros(len(A) - p)
            for k in range(1, p + 1):
                A_minus = A_minus + A[p - k:len(A) - k] * z_p[k - 1]
            A_plus = A[p:]
            d_y = d_y + __likelihood_d_y(x, q, A_plus, A_minus)
    return d_y


def cdarn_simulation(p, Q, c, Y, B, n, s, model_cross='neighbors', z_p='uniform'):
    """
    Simulate a temporal network following the :math:`CDARN(p)` model.

    The :math:`CDARN(p)` model  [1]_ is a generalization of the :math:`DARN(p)` model where also cross-correlations between links are taken into account.

    For a temporal network described by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}` the :math:`CDARN(p)` model describes the state :math:`A^{t}_{ij}` of the link :math:`(i,j)` at time :math:`t` with the following copying mechanism: (i) with probability :math:`q_{ij}`, the link is copied from a past state of a link in the backbone :math:`B` chosen by an assigned probability distribution, (ii) then the lag is selected with probability :math:`z_i` with :math:`i=1,\\ldots,p`, where :math:`p` represents the order of the memory, (iii) while with probability :math:`1-q_{ij}`, it is sampled with (marginal) probability :math:`y_{ij}`.

    In other words, the state :math:`A_{ij}^t` of the link :math:`(i,j)` is described by the following stochastic process at discrete time

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}A^{t}_{ij}= Q^{t}_{ij}  A^{t-Z_p^{t}}_{M_{ij}^t} + (1-Q^{t}_{ij}) Y^{t}_{ij}`

    with :math:`Q_{ij}^t\\sim B(q_{ij})` Bernoulli variable, :math:`M_{ij}` random variable picking values between the links in the backbone of :math:`(i,j)`,  :math:`Z_p^t` is a random variable picking values from :math:`1` to :math:`p` with probability :math:`z_i` :math:`i=1,\\ldots,p`, respectively, and :math:`Y_{ij}^t\\sim B(y_{ij})` is a Bernoulli variable.

    The random variable :math:`M_{ij}^t` depends both on the coupling model and the constant :math:`c`:

    1) Local Cross Correlation (LCC): :math:`M_{ij}^t` picks :math:`(i,j)` with probability :math:`1-c` and any other link in the neighbourhood :math:`\\partial B_{ij}` of :math:`(i,j)` with probability :math:`\\frac{c}{|\\partial B_{ij}|}`.

    2) Uniform Cross Correlation (UCC):  :math:`M_{ij}^t` picks :math:`(i,j)` with probability :math:`1-c` and any other link in the backbone :math:`B` of :math:`(i,j)` with probability :math:`\\frac{c}{|B-1|}`.

    Here :math:`|\\partial B_{ij}|` and :math:`|B-1|` denote the number of links adjacent to :math:`(i,j)` and the number of links in the backbone different from :math:`(i, j)`, respectively.


    Parameters
    __________

    p: integer
        Markov order
    Q: array_like
        Symmetric matrix. It represents the link-specific probability of copying from the past.
        A float input is accepted in the case of homogeneous copying probabilities for each link.
    c: float
        cross correlation parameter.
    Y: array_like
        Symmetric matrix. Bernoulli marginal probabilities.
        A float input is accepted in the case of homogeneous marginal probabilities for each link.
    B: array_like
        Symmetric matrix. Backbone of the temporal graph.
    n: integer
        Number of nodes in the graph.
    s: integer
        Temporal Network sample size, *i.e.* the length of the time series of adjacency matrices.
    model_cross: string, optional
        Type of cross correlation. When ``'neighbors'`` links are coupled to all other neighbouring
        links in the network backbone with equal strength (LCC), when ``'all'`` links are coupled to all other links in the backbone with equal strength (UCC). Default ``'neighbors'``.
    z_p: array/string, optional
        stochastic array of length :math:`p` representing the memory distribution \
        :math:`Z_p`. Default: ``'uniform'``.
        Possible strings:

        * ``'uniform'``: :math:`Z_p = (\\frac{1}{p}, \\dots, \\frac{1}{p}`),

        * ``'exponential'``: :math:`Z_p = \\frac{(1, e^{-1}, \\dots, e^{-p+1})}{\\left|\\left|(1, e^{-1}, \\dots, e^{-p+1}\\right|\\right|_2)}`

        * ``'normal'``: let :math:`f(x)=\\frac{e^{-(x-1)^2/2}}{\\sqrt{2\\pi}}` then :math:`Z_p = \\frac{(f(0), \\dots, f(p-1))}{\\left|\\left|(f(0), \\dots, f(p-1))\\right|\\right|}`


    :return: **simulation**: (list)
        Temporal network produced by a :math:`CDARN(p)` model.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Define input parameters

    .. code:: python

     >>>    p = 3
     >>>    Q = (np.ones((n, n)) - np.diag(np.ones(n))) * 0.5
     >>>    c = 0.4
     >>>    Y = (np.ones((n, n)) - np.diag(np.ones(n)))  * 0.3
     >>>    B = (np.ones((n, n)) - np.diag(np.ones(n)))
     >>>    n = 50
     >>>    s = 100


    Simulate the temporal network

     .. code:: python

     >>>    time_series = sm.cdarn_simulation(p, Q, c, Y, B, n, s)


    References
    ----------

    .. [1] Williams, O.E., Mazzarisi, P., Lillo, F. and Latora, V., 2021.
           Non-Markovian temporal networks with auto- and cross-correlated link dynamics.
           https://arxiv.org/abs/1909.08134
    """
    if isinstance(z_p, str):
        if z_p == 'uniform':
            z_p = np.ones(p) / p
        elif z_p == 'exponential':
            z_p = np.exp(-np.array(range(p))) / np.sum(np.exp(-np.array(range(p))))
        elif z_p == 'normal':
            z_p = normal.pdf(np.array(range(p)), 2, 1) / np.sum(normal.pdf(np.array(range(p)), 2, 1))
        else:
            raise ValueError('The admissible strings for z_p are ' " 'uniform' "', ' " 'exponential'" ' or ' " 'normal'"
                             ', otherwise use arrays' '.')

    if isinstance(Q, float):
        Q = np.ones(n, n) * Q
    if isinstance(Y, float):
        Y = np.ones(n, n) * Y

    simulation = []

    for k in range(p):
        x = choice(np.array([0, 1]), (n, n), p=[1 / 2, 1 / 2])
        x = np.array(x)
        simulation.append(x)
    for k in range(s):
        simulation.append(np.zeros([n, n]))
        for i in range(n):  # Upper triangular part
            for j in range(i + 1, n):
                if B[i, j] == 1:
                    if model_cross == 'neighbors':
                        B_ij = np.zeros((n, n))
                        B_ij[i, :] = B[i, :]
                        B_ij[:, j] = B[:, j]
                        B_ij = B_ij + np.transpose(B_ij)
                        B_ij[i, j] = 0
                        B_ij[j, i] = 0
                        b_ij = B_ij[np.triu_indices(n, 1)]  # transform upper triangle part to vector
                        neighbors = np.nonzero(b_ij)[0]
                        n_neighbors_ij = sum(B[:, i]) + sum(B[j, :]) - 2 * B[j, i]
                        auto_past_series = np.zeros(p)
                        if n_neighbors_ij != 0:
                            chosen_link = choice(list(neighbors), 1)
                            cross_past_series = np.zeros(p)
                            for past in range(p):
                                cross_past_series[past] = simulation[past + k][np.triu_indices(n, 1)][chosen_link]
                                auto_past_series[past] = simulation[past + k][i, j]
                            cross_past_element = choice(cross_past_series, 1, p=z_p)
                            auto_past_element = choice(auto_past_series, 1, p=z_p)
                            simulation[p + k][i, j] = choice([cross_past_element[0], auto_past_element[0], 1, 0], 1,
                                                             p=[Q[i, j] * c, Q[i, j] * (1 - c),
                                                                (1 - Q[i, j]) * Y[i, j],
                                                                (1 - Q[i, j]) * (1 - Y[i, j])])
                            simulation[p + k][j, i] = simulation[p + k][i, j]
                        else:
                            for past in range(p):
                                auto_past_series[past] = simulation[past + k][i, j]
                            auto_past_element = choice(auto_past_series, 1, p=z_p)
                            simulation[p + k][i, j] = choice([auto_past_element[0], 1, 0], 1,
                                                             p=[Q[i, j], (1 - Q[i, j]) * Y[i, j],
                                                                (1 - Q[i, j]) * (1 - Y[i, j])])
                            simulation[p + k][j, i] = simulation[p + k][i, j]
                    elif model_cross == 'all':
                        B_temp = np.array(B)
                        B_temp[i, j] = 0
                        B_upper = B_temp[np.triu_indices(n, 1)]
                        if sum(B_upper) == 0:
                            auto_past_series = np.zeros(p)
                            for past in range(p):
                                auto_past_series[past] = simulation[past + k][i, j]
                            auto_past_element = choice(auto_past_series, 1, p=z_p)
                            simulation[p + k][i, j] = choice([auto_past_element[0], 1, 0], 1,
                                                             p=[Q[i, j], (1 - Q[i, j]) * Y[i, j],
                                                                (1 - Q[i, j]) * (1 - Y[i, j])])
                            simulation[p + k][j, i] = simulation[p + k][i, j]
                        else:
                            chosen_link = choice(list(range(int((n * (n - 1)) / 2))), 1, p=B_upper / sum(B_upper))
                            auto_past_series = np.zeros(p)
                            cross_past_series = np.zeros(p)
                            for past in range(p):
                                auto_past_series[past] = simulation[past + k][i, j]
                                cross_past_series[past] = simulation[past + k][np.triu_indices(n, 1)][chosen_link]
                            auto_past_element = choice(auto_past_series, 1, p=z_p)
                            cross_past_element = choice(cross_past_series, 1, p=z_p)
                            simulation[p + k][i, j] = choice([cross_past_element[0], auto_past_element[0], 1, 0], 1,
                                                             p=[Q[i, j] * c, Q[i, j] * (1 - c),
                                                                (1 - Q[i, j]) * Y[i, j],
                                                                (1 - Q[i, j]) * (1 - Y[i, j])])
                            simulation[p + k][j, i] = simulation[p + k][i, j]
                    elif model_cross == 'itself':
                        auto_past_series = np.zeros(p)
                        for past in range(p):
                            auto_past_series[past] = simulation[past + k][i, j]
                        auto_past_element = choice(auto_past_series, 1, p=z_p)
                        simulation[p + k][i, j] = choice([auto_past_element[0], 1, 0], 1,
                                                         p=[Q[i, j], (1 - Q[i, j]) * Y[i, j],
                                                            (1 - Q[i, j]) * (1 - Y[i, j])])
                        simulation[p + k][j, i] = simulation[p + k][i, j]
    simulation = simulation[p:]
    return simulation


def cdarn(time_series, p, B, z_p='uniform', model_cross='neighbors', q='global', y='global', kernel=None,
          bandwidth=None):
    """
    Simulate a temporal network following the :math:`CDARN(p)` model.

    The :math:`CDARN(p)` model  [1]_ is a generalization of the :math:`DARN(p)` model where also cross-correlations between links are taken into account.

    For a temporal network described by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}` the :math:`CDARN(p)` model describes the state :math:`A^{t}_{ij}` of the link :math:`(i,j)` at time :math:`t` with the following copying mechanism: (i) with probability :math:`q_{ij}`, the link is copied from a past state of a link in the backbone :math:`B` chosen by an assigned probability distribution, (ii) then the lag is selected with probability :math:`z_i` with :math:`i=1,\\ldots,p`, where :math:`p` represents the order of the memory, (iii) while with probability :math:`1-q_{ij}`, it is sampled with (marginal) probability :math:`y_{ij}`.

    In other words, the state :math:`A_{ij}^t` of the link :math:`(i,j)` is described by the following stochastic process at discrete time

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}A^{t}_{ij}= Q^{t}_{ij}  A^{t-Z_p^{t}}_{M_{ij}^t} + (1-Q^{t}_{ij}) Y^{t}_{ij}`

    with :math:`Q_{ij}^t\\sim B(q_{ij})` Bernoulli variable, :math:`M_{ij}` random variable picking values between the links in the backbone of :math:`(i,j)`,  :math:`Z_p^t` is a random variable picking values from :math:`1` to :math:`p` with probability :math:`z_i` :math:`i=1,\\ldots,p`, respectively, and :math:`Y_{ij}^t\\sim B(y_{ij})` is a Bernoulli variable.

    The random variable :math:`M_{ij}^t` depends both on the coupling model and the constant :math:`c`:

    1) Local Cross Correlation (LCC): :math:`M_{ij}^t` picks :math:`(i,j)` with probability :math:`1-c` and any other link in the neighbourhood :math:`\\partial B_{ij}` of :math:`(i,j)` with probability :math:`\\frac{c}{|\\partial B_{ij}|}`.

    2) Uniform Cross Correlation (UCC):  :math:`M_{ij}^t` picks :math:`(i,j)` with probability :math:`1-c` and any other link in the backbone :math:`B` of :math:`(i,j)` with probability :math:`\\frac{c}{|B-1|}`.

    Here :math:`|\\partial B_{ij}|` and :math:`|B-1|` denote the number of links adjacent to :math:`(i,j)` and the number of links in the backbone different from :math:`(i, j)`, respectively.

    In the case of local likelihood estimation, the (global) parameters are time-varying, *i.e.* :math:`\\{q^t,c^t,y^t\\}_{t=1,\\ldots,s}`.


    Parameters
    __________
    time_series: List object
        List of symmetric matrices [:math:`A_0, A_1, \\dots, A_T`].
    p: integer
        Markov order.
    B: array_like
        Symmetric matrix. Backbone of the temporal graph.
    z_p: array/string, optional
        stochastic array of length :math:`p` representing the memory distribution \
        :math:`Z_p`. Default: ``'uniform'``.
        Possible strings:

        * ``'uniform'``: :math:`Z_p = (\\frac{1}{p}, \\dots, \\frac{1}{p}`),

        * ``'exponential'``: :math:`Z_p = \\frac{(1, e^{-1}, \\dots, e^{-p+1})}{\\left|\\left|(1, e^{-1}, \\dots, e^{-p+1}\\right|\\right|_2)}`

        * ``'normal'``: let :math:`f(x)=\\frac{e^{-(x-1)^2/2}}{\\sqrt{2\\pi}}` then :math:`Z_p = \\frac{(f(0), \\dots, f(p-1))}{\\left|\\left|(f(0), \\dots, f(p-1))\\right|\\right|}`
    model_cross: string, optional
        Type of cross correlation. When ``'neighbors'`` links are coupled to all other neighbouring
        links in the network backbone with equal strength (LCC), when ``'all'`` links are coupled to all other links in the backbone with equal strength (UCC). Default: ``'neighbors'``.
    q: string, optional
        Probability of copying from the past. If ``'global'`` the probability does not depend on the link, if ``'local'`` it depends on the link. Default: ``'global'``.
    y: string, optional
        Link density. If ``'global'`` the density does not depend on the link, if ``'local'`` it depends on the link. Default: ``'global'``.
    kernel: string, optional
        If not None parameters are time dependent. Default: None.
        Possible values: ``'epanechnikov'``, ``'gaussian'``.
    bandwidth: integer, optional
        Needed only when kernel is not None; bandwidth of the kernel.

    Returns
    _______

    q: float/array_like
        Probability parameter of copying from the past. Float when :math:`q` is ``'global'`` and time independent, array when is ``'global'`` and time dependent and symmetric matrix when :math:`q` is ``'local'``.
    c: float/array_like
        Cross correlation parameter. Float when :math:`c` is time independent, array when is time dependent.
    y: float/array_like
        Link density parameter. Float when :math:`y` is ``'global'`` and time independent, array when is ``'global'``\
         and time dependent and symmetric matrix when :math:`y` is ``'local'``.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Create temporal network

    .. code:: python

     >>>    p = 3
     >>>    Q = (np.ones((n, n)) - np.diag(np.ones(n))) * 0.5
     >>>    c = 0.4
     >>>    Y = (np.ones((n, n)) - np.diag(np.ones(n)))  * 0.3
     >>>    B = (np.ones((n, n)) - np.diag(np.ones(n)))
     >>>    n = 50
     >>>    s = 100
     >>>    time_series = sm.cdarn_simulation(p, Q, c, Y, B, n, s)


    Estimate the :math:`CDARN(p)` model parameters

    >>>    q, c, y = sm.cdarn(time_series, p, B)

     .. code:: python


    References
    __________

    .. [1] Williams, O.E., Mazzarisi, P., Lillo, F. and Latora, V., 2021.
           Non-Markovian temporal networks with auto- and cross-correlated link dynamics.
           https://arxiv.org/abs/1909.08134

    """

    n = np.shape(time_series[0])[0]
    T = len(time_series)

    # convergence parameters
    tol = 1e-7
    maxit = 10 ** 3

    # set memory kernel
    if isinstance(z_p, str):
        if z_p == 'uniform':
            z_p = np.ones((p, 1)) / p
        elif z_p == 'exponential':
            z_p = np.exp(-np.array(range(p))) / np.sum(np.exp(-np.array(range(p))))
        elif z_p == 'normal':
            z_p = normal.pdf(np.array(range(p)), 2, 1) / np.sum(normal.pdf(np.array(range(p)), 2, 1))
        else:
            raise ValueError('The value of z_p must be ' " 'uniform' "'  or ' " 'exponential'" ' or ' " 'normal'" '.')

    if q == 'global' and y == 'global':
        if kernel is None:
            # set cross correlation
            B = B - np.diag(np.diag(B))
            b = B[np.triu_indices(n, 1)]
            S = int(sum(b))
            if S == 0:
                raise ValueError('There are no links in the backbone.')
            elif S == 1:
                raise ValueError('There is only 1 link in the backbone.')
            else:
                L = np.zeros((n, n))
                if model_cross == 'neighbors':
                    L = np.zeros([n * (n - 1) // 2, n * (n - 1) // 2])
                    temp = 0
                    for i in range(n - 1):
                        for j in range(i + 1, n):
                            B_ij = np.zeros((n, n))
                            B_ij[i, :] = B[i, :]
                            B_ij[:, j] = B[:, j]
                            B_ij = B_ij + np.transpose(B_ij)
                            b_ij = B_ij[np.triu_indices(n, 1)]  # transform upper triangle part to vector
                            L[temp, :] = b_ij
                            temp = temp + 1
                    L = np.delete(L, b == 0, 0)
                    L = np.delete(L, b == 0, 1)
                    L = L - np.diag(np.diag(L))
                    for k in range(S):
                        if sum(L[k, :]) != 0:
                            L[k, :] = L[k, :] / sum(L[k, :])
                if model_cross == 'all':
                    L = (np.ones((S, S)) - np.identity(S)) / (S - 1)
                if model_cross == 'itself':
                    L = np.zeros([S, S])

                # vectorization of time series
                X = np.zeros((n * (n - 1) // 2, T))
                for t in range(T):
                    X[:, t] = time_series[t][np.triu_indices(n, 1)]
                X = X[b != 0, :]

                # X is S x T now

                dit = np.zeros((S, T))
                for shift in range(p):
                    dit = dit + z_p[shift] * (X == np.roll(X, shift + 1, 1))

                cit = np.zeros((S, T))
                for i in range(S):
                    for j in range(S):
                        for shift in range(p):
                            cit[i, :] = cit[i, :] + L[i, j] * z_p[shift] * (X[i, :] == np.roll(X, shift + 1, 1)[j, :])

                # metrics for maximum likelihood estimation
                X = X[:, p:]
                dit = dit[:, p:]
                cit = cit[:, p:]
                delta1it = X
                delta0it = 1 * (X == 0)

                b_0 = 0.001
                b_1 = 0.999

                if model_cross != 'itself':
                    # naive y_0
                    y_0 = sum(sum(X)) / ((T - p) * S)
                    q_0 = 0.1 + 0.8 * np.random.rand(1)
                    a_0 = 0.1 + 0.8 * np.random.rand(1)

                    precision = 1
                    temp_ite = 1

                    while (precision > tol) and (temp_ite <= maxit):
                        marginal_0 = (y_0 ** X) * ((1 - y_0) ** (1 - X))

                        # inference of q

                        r_0 = cdarn_likelihood_q(b_0, a_0, dit, cit, marginal_0)
                        r_1 = cdarn_likelihood_q(b_1, a_0, dit, cit, marginal_0)
                        if r_0 * r_1 < 0:
                            q_1 = root_scalar(cdarn_likelihood_q, args=(a_0, dit, cit, marginal_0),
                                              bracket=(b_0, b_1)).root
                        else:
                            q_1 = 0.1 + 0.8 * np.random.rand(1)

                        # inference of a

                        f0 = cdarn_likelihood_a(b_0, q_1, dit, cit, marginal_0)
                        f1 = cdarn_likelihood_a(b_1, q_1, dit, cit, marginal_0)
                        if f0 * f1 < 0:
                            a_1 = root_scalar(cdarn_likelihood_a, args=(q_1, dit, cit, marginal_0),
                                              bracket=(b_0, b_1)).root
                        else:
                            a_1 = 0.1 + 0.8 * np.random.rand(1)

                        # inference of y
                        chi_0 = cdarn_likelihood_y(b_0, delta1it, delta0it, q_1, a_1, dit, cit, X)
                        chi_1 = cdarn_likelihood_y(b_1, delta1it, delta0it, q_1, a_1, dit, cit, X)
                        if chi_0 * chi_1 < 0:
                            y_1 = root_scalar(cdarn_likelihood_y, args=(delta1it, delta0it, q_1, a_1, dit, cit, X),
                                              bracket=(b_0, b_1)).root
                        else:
                            y_1 = y_0

                        # update variables
                        precision = max([abs((y_1 - y_0) / y_0), abs((q_1 - q_0) / q_0), abs((a_1 - a_0) / a_0)])
                        y_0 = y_1
                        q_0 = q_1
                        a_0 = a_1
                        temp_ite = temp_ite + 1

                else:
                    # naive y_0
                    y_0 = sum(sum(X)) / ((T - p) * L)
                    q_0 = 0.1 + 0.9 * np.random.rand(1)
                    a_0 = 1

                    precision = 1
                    temp_ite = 1

                    while (precision > tol) and (temp_ite <= maxit):
                        marginal_0 = (y_0 ** X) * (1 - y_0) ** (1 - X)
                        # inference of q
                        r_0 = cdarn_likelihood_q(b_0, a_0, dit, cit, marginal_0)
                        r_1 = cdarn_likelihood_q(b_1, a_0, dit, cit, marginal_0)
                        if r_0 * r_1 < 0:
                            q_1 = root_scalar(cdarn_likelihood_q, args=(a_0, dit, cit, marginal_0),
                                              bracket=(b_0, b_1)).root
                        else:
                            q_1 = 0.1 + 0.9 * np.rand.random(1)

                        a_1 = 1

                        # inference of y
                        chi_0 = cdarn_likelihood_y(b_0, delta1it, delta0it, q_1, a_1, dit, cit, X)
                        chi_1 = cdarn_likelihood_y(b_1, delta1it, delta0it, q_1, a_1, dit, cit, X)
                        if chi_0 * chi_1 < 0:
                            y_1 = root_scalar(cdarn_likelihood_y, args=(delta1it, delta0it, q_1, a_1, dit, cit, X),
                                              bracket=(b_0, b_1)).root
                        else:
                            y_1 = y_0

                        # update variables

                        precision = max([abs((y_1 - y_0) / y_0), abs((q_1 - q_0) / q_0)])
                        y_0 = y_1
                        q_0 = q_1
                        a_0 = a_1
                        temp_ite = temp_ite + 1
                if temp_ite > maxit:
                    raise Warning("Convergence not achieved.")
                y = y_0
                q = q_0
                a = a_0
                c = 1 - a
                return q, c, y

        # temporal parameters

        else:
            # set cross correlation
            B = B - np.diag(np.diag(B))
            b = B[np.triu_indices(n, 1)]
            S = int(sum(b))
            if S == 0:
                raise ValueError('There are no links in the backbone.')
            elif S == 1:
                raise ValueError('There is only 1 link in the backbone.')
            else:
                L = np.zeros((n, n))
                if model_cross == 'neighbors':
                    L = np.zeros([n * (n - 1) // 2, n * (n - 1) // 2])
                    temp = 0
                    for i in range(n - 1):
                        for j in range(i + 1, n):
                            B_ij = np.zeros((n, n))
                            B_ij[i, :] = B[i, :]
                            B_ij[:, j] = B[:, j]
                            B_ij = B_ij + np.transpose(B_ij)
                            b_ij = B_ij[np.triu_indices(n, 1)]  # transform upper triangle part to vector
                            L[temp, :] = b_ij
                            temp = temp + 1
                    L = np.delete(L, b == 0, 0)
                    L = np.delete(L, b == 0, 1)
                    L = L - np.diag(np.diag(L))
                    for k in range(S):
                        if sum(L[k, :]) != 0:
                            L[k, :] = L[k, :] / sum(L[k, :])
                if model_cross == 'all':
                    L = (np.ones((S, S)) - np.identity(S)) / (S - 1)
                if model_cross == 'itself':
                    L = np.zeros([S, S])

                # vectorization of time series
                X = np.zeros((n * (n - 1) // 2, T))
                for t in range(T):
                    X[:, t] = time_series[t][np.triu_indices(n, 1)]
                X = X[b != 0, :]

                # X is S x T now

                dit = np.zeros((S, T))
                for shift in range(p):
                    dit = dit + z_p[shift] * (X == np.roll(X, shift + 1, 1))

                cit = np.zeros((S, T))
                for i in range(S):
                    for j in range(S):
                        for shift in range(p):
                            cit[i, :] = cit[i, :] + L[i, j] * z_p[shift] * (X[i, :] == np.roll(X, shift + 1, 1)[j, :])

                # metrics for maximum likelihood estimation
                X = X[:, p:]
                dit = dit[:, p:]
                cit = cit[:, p:]
                delta1it = X
                delta0it = 1 * (X == 0)

                b_0 = 0.001
                b_1 = 0.999

                y_t = []
                q_t = []
                c_t = []
                # naive y_0
                y_0 = sum(sum(X)) / ((T - p) * S)
                q_0 = 0.1 + 0.8 * np.random.rand(1)
                a_0 = 0.1 + 0.8 * np.random.rand(1)
                if bandwidth is None:
                    bandwidth = min(40, max(T // 20, 20))
                for t in range(bandwidth, T - p - bandwidth):

                    precision = 1
                    temp_ite = 1

                    while (precision > tol) and (temp_ite <= maxit):
                        marginal_0 = (y_0 ** X) * ((1 - y_0) ** (1 - X))

                        # inference of Q
                        r_0 = cdarn_likelihood_q_temporal(b_0, a_0, dit, cit, marginal_0, bandwidth, t, kernel)
                        r_1 = cdarn_likelihood_q_temporal(b_1, a_0, dit, cit, marginal_0, bandwidth, t, kernel)
                        if r_0 * r_1 < 0:
                            q_1 = root_scalar(cdarn_likelihood_q_temporal,
                                              args=(a_0, dit, cit, marginal_0, bandwidth, t, kernel),
                                              bracket=(b_0, b_1)).root
                        else:
                            q_1 = 0.1 + 0.8 * np.random.rand(1)

                        # inference of a

                        f0 = cdarn_likelihood_a_temporal(b_0, q_1, dit, cit, marginal_0, bandwidth, t, kernel)
                        f1 = cdarn_likelihood_a_temporal(b_1, q_1, dit, cit, marginal_0, bandwidth, t, kernel)
                        if f0 * f1 < 0:
                            a_1 = root_scalar(cdarn_likelihood_a_temporal, args=(q_1, dit, cit, marginal_0,
                                                                                 bandwidth, t, kernel),
                                              bracket=(b_0, b_1)).root
                        else:
                            a_1 = 0.1 + 0.8 * np.random.rand(1)

                        # inference of y
                        chi_0 = cdarn_likelihood_y_temporal(b_0, delta1it, delta0it, q_1, a_1, dit, cit, X, bandwidth,
                                                            t, kernel)
                        chi_1 = cdarn_likelihood_y_temporal(b_1, delta1it, delta0it, q_1, a_1, dit, cit, X, bandwidth,
                                                            t, kernel)
                        if chi_0 * chi_1 < 0:
                            y_1 = root_scalar(cdarn_likelihood_y_temporal,
                                              args=(delta1it, delta0it, q_1, a_1, dit, cit, X, bandwidth, t, kernel),
                                              bracket=(b_0, b_1)).root
                        else:
                            y_1 = y_0

                        # update variables
                        precision = max([abs((y_1 - y_0) / y_0), abs((q_1 - q_0) / q_0), abs((a_1 - a_0) / a_0)])
                        y_0 = y_1
                        q_0 = q_1
                        a_0 = a_1
                        temp_ite = temp_ite + 1

                    if temp_ite > maxit:
                        raise Warning("Convergence not achieved.")
                    c_0 = 1 - a_0
                    y_t.append(y_0)
                    q_t.append(q_0)
                    c_t.append(c_0)

                return q_t, c_t, y_t

    if q == 'local' and y == 'global':
        # set cross correlation
        B = B - np.diag(np.diag(B))
        b = B[np.triu_indices(n, 1)]
        S = int(sum(b))
        if S == 0:
            raise ValueError('There are no links in the backbone.')
        elif S == 1:
            raise ValueError('There is only 1 link in the backbone.')
        else:
            L = np.zeros((n, n))
            if model_cross == 'neighbors':
                L = np.zeros([n * (n - 1) // 2, n * (n - 1) // 2])
                temp = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        B_ij = np.zeros((n, n))
                        B_ij[i, :] = B[i, :]
                        B_ij[:, j] = B[:, j]
                        B_ij = B_ij + np.transpose(B_ij)
                        b_ij = B_ij[np.triu_indices(n, 1)]  # transform upper triangle part to vector
                        L[temp, :] = b_ij
                        temp = temp + 1
                L = np.delete(L, b == 0, 0)
                L = np.delete(L, b == 0, 1)
                L = L - np.diag(np.diag(L))
                for k in range(S):
                    if sum(L[k, :]) != 0:
                        L[k, :] = L[k, :] / sum(L[k, :])
            if model_cross == 'all':
                L = (np.ones((S, S)) - np.identity(S)) / (S - 1)
            if model_cross == 'itself':
                L = np.zeros([S, S])

            # vectorization of time series
            X = np.zeros((n * (n - 1) // 2, T))
            for t in range(T):
                X[:, t] = time_series[t][np.triu_indices(n, 1)]
            X = X[b != 0, :]

            # X is S x T now

            dit = np.zeros((S, T))
            for shift in range(p):
                dit = dit + z_p[shift] * (X == np.roll(X, shift + 1, 1))

            cit = np.zeros((S, T))
            for i in range(S):
                for j in range(S):
                    for shift in range(p):
                        cit[i, :] = cit[i, :] + L[i, j] * z_p[shift] * (X[i, :] == np.roll(X, shift + 1, 1)[j, :])

            # metrics for maximum likelihood estimation
            X = X[:, p:]
            dit = dit[:, p:]
            cit = cit[:, p:]
            delta1it = X
            delta0it = 1 * (X == 0)

            b_0 = 0.001
            b_1 = 0.999

            # naive y_0
            y_0 = sum(sum(X)) / ((T - p) * S)
            Q_0 = (0.1 + 0.8 * np.random.rand(S))
            Q_1 = np.zeros(S)
            a_0 = 0.1 + 0.8 * np.random.rand(1)

            precision = 1
            temp_ite = 1

            while (precision > tol) and (temp_ite <= maxit):
                marginal_0 = (y_0 ** X) * ((1 - y_0) ** (1 - X))

                # inference of Q
                for link in range(S):
                    r_0 = cdarn_likelihood_q_local_q(b_0, a_0, dit[link], cit[link], marginal_0[link])
                    r_1 = cdarn_likelihood_q_local_q(b_1, a_0, dit[link], cit[link], marginal_0[link])
                    if r_0 * r_1 < 0:
                        Q_1[link] = root_scalar(cdarn_likelihood_q_local_q,
                                                args=(a_0, dit[link], cit[link], marginal_0[link]),
                                                bracket=(b_0, b_1)).root
                    else:
                        Q_1[link] = 0.1 + 0.8 * np.random.rand(1)

                # inference of a

                f0 = cdarn_likelihood_a_local_q(b_0, Q_1, dit, cit, marginal_0)
                f1 = cdarn_likelihood_a_local_q(b_1, Q_1, dit, cit, marginal_0)
                if f0 * f1 < 0:
                    a_1 = root_scalar(cdarn_likelihood_a_local_q, args=(Q_1, dit, cit, marginal_0),
                                      bracket=(b_0, b_1)).root
                else:
                    a_1 = 0.1 + 0.8 * np.random.rand(1)

                # inference of y
                chi_0 = cdarn_likelihood_y_local_q(b_0, delta1it, delta0it, Q_1, a_1, dit, cit, X)
                chi_1 = cdarn_likelihood_y_local_q(b_1, delta1it, delta0it, Q_1, a_1, dit, cit, X)
                if chi_0 * chi_1 < 0:
                    y_1 = root_scalar(cdarn_likelihood_y_local_q, args=(delta1it, delta0it, Q_1, a_1, dit, cit, X),
                                      bracket=(b_0, b_1)).root
                else:
                    y_1 = y_0

                # update variables
                precision = max([abs((y_1 - y_0) / y_0), max(abs((Q_1 - Q_0) / Q_0)), abs((a_1 - a_0) / a_0)])
                y_0 = y_1
                Q_0 = Q_1
                a_0 = a_1
                temp_ite = temp_ite + 1

        if temp_ite > maxit:
            raise Warning("Convergence not achieved.")

        y = y_0
        Q = np.zeros((n, n))
        b[b == 1] = Q_0
        Q[np.triu_indices(n, 1)] = b
        Q = Q + Q.transpose()
        a = a_0
        c = 1 - a
        return Q, c, y

    elif q == 'global' and y == 'local':
        # set cross correlation
        B = B - np.diag(np.diag(B))
        b = B[np.triu_indices(n, 1)]
        S = int(sum(b))
        if S == 0:
            raise ValueError('There are no links in the backbone.')
        elif S == 1:
            raise ValueError('There is only 1 link in the backbone.')
        else:
            L = np.zeros((n, n))
            if model_cross == 'neighbors':
                L = np.zeros([n * (n - 1) // 2, n * (n - 1) // 2])
                temp = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        B_ij = np.zeros((n, n))
                        B_ij[i, :] = B[i, :]
                        B_ij[:, j] = B[:, j]
                        B_ij = B_ij + np.transpose(B_ij)
                        b_ij = B_ij[np.triu_indices(n, 1)]  # transform upper triangle part to vector
                        L[temp, :] = b_ij
                        temp = temp + 1
                L = np.delete(L, b == 0, 0)
                L = np.delete(L, b == 0, 1)
                L = L - np.diag(np.diag(L))
                for k in range(S):
                    if sum(L[k, :]) != 0:
                        L[k, :] = L[k, :] / sum(L[k, :])
            if model_cross == 'all':
                L = (np.ones((S, S)) - np.identity(S)) / (S - 1)
            if model_cross == 'itself':
                L = np.zeros([S, S])

            # vectorization of time series
            X = np.zeros((n * (n - 1) // 2, T))
            for t in range(T):
                X[:, t] = time_series[t][np.triu_indices(n, 1)]
            X = X[b != 0, :]

            # X is S x T now

            dit = np.zeros((S, T))
            for shift in range(p):
                dit = dit + z_p[shift] * (X == np.roll(X, shift + 1, 1))

            cit = np.zeros((S, T))
            for i in range(S):
                for j in range(S):
                    for shift in range(p):
                        cit[i, :] = cit[i, :] + L[i, j] * z_p[shift] * (X[i, :] == np.roll(X, shift + 1, 1)[j, :])

            # metrics for maximum likelihood estimation
            X = X[:, p:]
            dit = dit[:, p:]
            cit = cit[:, p:]
            delta1it = X
            delta0it = 1 * (X == 0)

            b_0 = 0.001
            b_1 = 0.999

            # naive y_0
            Y_0 = sum(sum(X)) / ((T - p) * S) * np.ones(S)
            q_0 = 0.1 + 0.8 * np.random.rand(1)
            a_0 = 0.1 + 0.8 * np.random.rand(1)
            Y_1 = np.zeros(S)

            precision = 1
            temp_ite = 1

            while (precision > tol) and (temp_ite <= maxit):
                marginal_0 = np.zeros(S)
                for link in range(S):
                    marginal_0 = (Y_0[link] * X) + ((1 - Y_0[link]) * (1 - X))

                # inference of Q
                r_0 = cdarn_likelihood_q_local_y(b_0, a_0, dit, cit, marginal_0)
                r_1 = cdarn_likelihood_q_local_y(b_1, a_0, dit, cit, marginal_0)
                if r_0 * r_1 < 0:
                    q_1 = root_scalar(cdarn_likelihood_q_local_y,
                                      args=(a_0, dit, cit, marginal_0),
                                      bracket=(b_0, b_1)).root
                else:
                    q_1 = 0.1 + 0.8 * np.random.rand(1)

                # inference of a

                f0 = cdarn_likelihood_a_local_y(b_0, q_1, dit, cit, marginal_0)
                f1 = cdarn_likelihood_a_local_y(b_1, q_1, dit, cit, marginal_0)
                if f0 * f1 < 0:
                    a_1 = root_scalar(cdarn_likelihood_a_local_y, args=(q_1, dit, cit, marginal_0),
                                      bracket=(b_0, b_1)).root
                else:
                    a_1 = 0.1 + 0.8 * np.random.rand(1)

                # inference of y
                for link in range(S):
                    chi_0 = cdarn_likelihood_y_local_y(b_0, delta1it[link], delta0it[link], q_1, a_1, dit[link],
                                                       cit[link], X[link])
                    chi_1 = cdarn_likelihood_y_local_y(b_1, delta1it[link], delta0it[link], q_1, a_1, dit[link],
                                                       cit[link], X[link])
                    if chi_0 * chi_1 < 0:
                        Y_1[link] = root_scalar(cdarn_likelihood_y_local_y, args=(
                            delta1it[link], delta0it[link], q_1, a_1, dit[link], cit[link], X[link]),
                                                bracket=(b_0, b_1)).root
                    else:
                        Y_1[link] = Y_0[link]

                # update variables
                precision = max([max(abs((Y_1 - Y_0) / Y_0)), abs((q_1 - q_0) / q_0), abs((a_1 - a_0) / a_0)])
                Y_0 = Y_1
                q_0 = q_1
                a_0 = a_1
                temp_ite = temp_ite + 1

            if temp_ite > maxit:
                raise Warning("Convergence not achieved.")

            q = q_0
            Y = np.zeros((n, n))
            b[b == 1] = Y_0
            Y[np.triu_indices(n, 1)] = b
            Y = Y + Y.transpose()
            a = a_0
            c = 1 - a
            return q, c, Y

    elif q == 'local' and y == 'local':
        # set cross correlation
        B = B - np.diag(np.diag(B))
        b = B[np.triu_indices(n, 1)]
        S = int(sum(b))
        if S == 0:
            raise ValueError('There are no links in the backbone.')
        elif S == 1:
            raise ValueError('There is only 1 link in the backbone.')
        else:
            L = np.zeros((n, n))
            if model_cross == 'neighbors':
                L = np.zeros([n * (n - 1) // 2, n * (n - 1) // 2])
                temp = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        B_ij = np.zeros((n, n))
                        B_ij[i, :] = B[i, :]
                        B_ij[:, j] = B[:, j]
                        B_ij = B_ij + np.transpose(B_ij)
                        b_ij = B_ij[np.triu_indices(n, 1)]  # transform upper triangle part to vector
                        L[temp, :] = b_ij
                        temp = temp + 1
                L = np.delete(L, b == 0, 0)
                L = np.delete(L, b == 0, 1)
                L = L - np.diag(np.diag(L))
                for k in range(S):
                    if sum(L[k, :]) != 0:
                        L[k, :] = L[k, :] / sum(L[k, :])
            if model_cross == 'all':
                L = (np.ones((S, S)) - np.identity(S)) / (S - 1)
            if model_cross == 'itself':
                L = np.zeros([S, S])

            # vectorization of time series
            X = np.zeros((n * (n - 1) // 2, T))
            for t in range(T):
                X[:, t] = time_series[t][np.triu_indices(n, 1)]
            X = X[b != 0, :]

            # X is S x T now

            dit = np.zeros((S, T))
            for shift in range(p):
                dit = dit + z_p[shift] * (X == np.roll(X, shift + 1, 1))

            cit = np.zeros((S, T))
            for i in range(S):
                for j in range(S):
                    for shift in range(p):
                        cit[i, :] = cit[i, :] + L[i, j] * z_p[shift] * (X[i, :] == np.roll(X, shift + 1, 1)[j, :])

            # metrics for maximum likelihood estimation
            X = X[:, p:]
            dit = dit[:, p:]
            cit = cit[:, p:]
            delta1it = X
            delta0it = 1 * (X == 0)

            b_0 = 0.001
            b_1 = 0.999

            # naive y_0
            Y_0 = sum(sum(X)) / ((T - p) * S) * np.ones(S)
            Y_1 = np.zeros(S)
            Q_0 = (0.1 + 0.8 * np.random.rand(S))
            Q_1 = np.zeros(S)
            a_0 = 0.1 + 0.8 * np.random.rand(1)

            precision = 1
            temp_ite = 1

            while (precision > tol) and (temp_ite <= maxit):
                marginal_0 = np.zeros(S)
                for link in range(S):
                    marginal_0 = (Y_0[link] * X) + ((1 - Y_0[link]) * (1 - X))

                # inference of Q
                for link in range(S):
                    r_0 = cdarn_likelihood_q_local_qy(b_0, a_0, dit[link], cit[link], marginal_0[link])
                    r_1 = cdarn_likelihood_q_local_qy(b_1, a_0, dit[link], cit[link], marginal_0[link])
                    if r_0 * r_1 < 0:
                        Q_1[link] = root_scalar(cdarn_likelihood_q_local_qy,
                                                args=(a_0, dit[link], cit[link], marginal_0[link]),
                                                bracket=(b_0, b_1)).root
                    else:
                        Q_1[link] = 0.1 + 0.8 * np.random.rand(1)

                # inference of a

                f0 = cdarn_likelihood_a_local_qy(b_0, Q_1, dit, cit, marginal_0)
                f1 = cdarn_likelihood_a_local_qy(b_1, Q_1, dit, cit, marginal_0)
                if f0 * f1 < 0:
                    a_1 = root_scalar(cdarn_likelihood_a_local_qy, args=(Q_1, dit, cit, marginal_0),
                                      bracket=(b_0, b_1)).root
                else:
                    a_1 = 0.1 + 0.8 * np.random.rand(1)

                # inference of y
                for link in range(S):
                    chi_0 = cdarn_likelihood_y_local_qy(b_0, delta1it[link], delta0it[link], Q_1[link], a_1,
                                                        dit[link], cit[link], X[link])
                    chi_1 = cdarn_likelihood_y_local_qy(b_1, delta1it[link], delta0it[link], Q_1[link], a_1,
                                                        dit[link], cit[link], X[link])
                    if chi_0 * chi_1 < 0:
                        Y_1[link] = root_scalar(cdarn_likelihood_y_local_qy, args=(
                            delta1it[link], delta0it[link], Q_1[link], a_1, dit[link], cit[link], X[link]),
                                                bracket=(b_0, b_1)).root
                    else:
                        Y_1[link] = Y_0[link]

                # update variables
                precision = max([max(abs((Y_1 - Y_0) / Y_0)), max(abs((Q_1 - Q_0) / Q_0)), abs((a_1 - a_0) / a_0)])
                Y_0 = Y_1
                Q_0 = Q_1
                a_0 = a_1
                temp_ite = temp_ite + 1

            if temp_ite > maxit:
                raise Warning("Convergence not achieved.")

            Q = np.zeros((n, n))
            b_temp = b.copy()
            b_temp[b_temp == 1] = Q_0
            Q[np.triu_indices(n, 1)] = b_temp
            Y = np.zeros((n, n))
            b[b == 1] = Y_0
            Y[np.triu_indices(n, 1)] = b
            Y = Y + Y.transpose()
            a = a_0
            c = 1 - a
            return Q, c, Y


def cdarn_likelihood_q(q, a, dit, cit, marginal):
    d_q = sum(sum(((a * dit + (1 - a) * cit) - marginal) /
                  (q * (a * dit + (1 - a) * cit) + (1 - q) * marginal)))
    return d_q


def cdarn_likelihood_q_temporal(q, a, dit, cit, marginal, bandwidth, t, kernel):
    d_q_temporal = 0
    for k in range(-bandwidth, bandwidth):
        d_q_temporal = d_q_temporal + sum(((a * dit[:, t + k] + (1 - a) * cit[:, t + k]) - marginal[:, t + k]) /
                                          (q * (a * dit[:, t + k] + (1 - a) * cit[:, t + k]) + (1 - q)
                                           * marginal[:, t + k])) * temporal_kernel(bandwidth, t, t + k, kernel)
    return d_q_temporal


def cdarn_likelihood_q_local_q(q_link, a, dit, cit, marginal):
    d_q_link = sum(((a * dit + (1 - a) * cit) - marginal) /
                   (q_link * (a * dit + (1 - a) * cit) + (1 - q_link) * marginal))
    return d_q_link


def cdarn_likelihood_q_local_y(q, a, dit, cit, marginal):
    d_q_link = 0
    for link in range(len(marginal)):
        d_q_link = d_q_link + sum(((a * dit[link] + (1 - a) * cit[link]) - marginal[link]) /
                                  (q * (a * dit[link] + (1 - a) * cit[link]) + (1 - q) * marginal[link]))
    return d_q_link


def cdarn_likelihood_q_local_qy(q_link, a, dit, cit, marginal):
    d_q_link = sum(((a * dit + (1 - a) * cit) - marginal) /
                   (q_link * (a * dit + (1 - a) * cit) + (1 - q_link) * marginal))
    return d_q_link


def cdarn_likelihood_a(a, q, dit, cit, marginal):
    d_a = sum(sum((dit - cit) / (q * (a * dit + (1 - a) * cit) + (1 - q) * marginal)))
    return d_a


def cdarn_likelihood_a_temporal(a, q, dit, cit, marginal, bandwidth, t, kernel):
    d_a_temporal = 0
    for k in range(-bandwidth, bandwidth):
        d_a_temporal = d_a_temporal + sum((dit[:, t + k] - cit[:, t + k]) /
                                          (q * (a * dit[:, t + k] + (1 - a) * cit[:, t + k]) + (1 - q)
                                           * marginal[:, t + k])) * temporal_kernel(bandwidth, t, t + k, kernel)
    return d_a_temporal


def cdarn_likelihood_a_local_q(a, Q, dit, cit, marginal):
    d_a_link = 0
    for link in range(len(Q)):
        d_a_link = d_a_link + sum((dit[link] - cit[link]) /
                                  (Q[link] * (a * dit[link] + (1 - a) * cit[link]) + (1 - Q[link]) * marginal[link]))
    return d_a_link


def cdarn_likelihood_a_local_y(a, q, dit, cit, marginal):
    d_a_link = 0
    for link in range(len(marginal)):
        d_a_link = d_a_link + sum((dit[link] - cit[link]) /
                                  (q * (a * dit[link] + (1 - a) * cit[link]) + (1 - q) * marginal[link]))
    return d_a_link


def cdarn_likelihood_a_local_qy(a, q, dit, cit, marginal):
    d_a_link = 0
    for link in range(len(marginal)):
        d_a_link = d_a_link + sum((dit[link] - cit[link]) /
                                  (q[link] * (a * dit[link] + (1 - a) * cit[link]) + (1 - q[link]) * marginal[link]))
    return d_a_link


def cdarn_likelihood_y(y, delta1it, delta0it, q, a, dit, cit, X):
    d_y = sum(sum((delta1it - delta0it) / (q * (a * dit + (1 - a) * cit) + (1 - q) * bernoulli_y(y, X))))
    return d_y


def cdarn_likelihood_y_temporal(y, delta1it, delta0it, q, a, dit, cit, X, bandwidth, t, kernel):
    d_y_temporal = 0
    for k in range(-bandwidth, bandwidth):
        d_y_temporal = d_y_temporal + sum((delta1it[:, t + k] - delta0it[:, t + k]) /
                                          (q * (a * dit[:, t + k] +
                                                (1 - a) * cit[:, t + k]) + (1 - q) *
                                           bernoulli_y(y, X[:, t + k]))) * temporal_kernel(bandwidth, t, t + k, kernel)
    return d_y_temporal


def cdarn_likelihood_y_local_q(y, delta1it, delta0it, Q, a, dit, cit, X):
    d_y_link = 0
    for link in range(len(Q)):
        d_y_link = d_y_link + sum((delta1it[link] - delta0it[link]) /
                                  (Q[link] * (a * dit[link] + (1 - a) * cit[link]) + (1 - Q[link]) *
                                   bernoulli_y(y, X)[link]))
    return d_y_link


def cdarn_likelihood_y_local_y(y_link, delta1it, delta0it, q, a, dit, cit, X):
    d_y_link = sum((delta1it - delta0it) / (q * (a * dit + (1 - a) * cit) + (1 - q) * bernoulli_y(y_link, X)))
    return d_y_link


def cdarn_likelihood_y_local_qy(y_link, delta1it, delta0it, q, a, dit, cit, X):
    d_y_link = sum((delta1it - delta0it) / (q * (a * dit + (1 - a) * cit) + (1 - q) * bernoulli_y(y_link, X)))
    return d_y_link


def bernoulli_y(y, X):
    bernoulli = (y * X) + ((1 - y) * (1 - X))
    return bernoulli


def temporal_kernel(bandwidth, t, s, kernel):
    if kernel == 'epanechnikov':
        K = 1 - (abs(t - s) / bandwidth) ** 2
    elif kernel == 'gaussian':
        K = np.exp(-1 / 2 * ((t - s) / bandwidth) ** 2)
    else:
        raise ValueError('The value of kernel must be ' " 'epanechnikov' or ' " 'gaussian'"  "'.')
    return K


# FITNESS MODELS

def tgrg(time_series, tol=1e-2, maxit=1e2):
    """
    Estimate by an expectation-maximization algorithm the parameters of the Temporally Generalized Random Graph model (:math:`TGRG`).

    For the :math:`TGRG` model [1]_, with temporal network is represented by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}`, each node :math:`i` is described by a latent variable :math:`\\theta_i^t`, namely the fitness of the node, which evolves over time by following a covariance stationary autoregressive process :math:`AR(1)`:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\theta_i^t = \\varphi_{0,i} + \\varphi_{1,i} \\theta_i^{t-1} + \\epsilon_i^t`,

    with :math:`\\varphi_{0,i}\\in \\mathbb{R}`, :math:`|\\varphi_{1,i}|<1`, and i.i.d. normal innovations :math:`\\epsilon_i^t\\sim \\mathcal{N}(0, \\sigma_i^2)`.

    Then, the observation equation for the network snapshot at time :math:`t` is given by :math:`\\binom{N}{2}` independent Bernoulli trials whose conditional probability is:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\mathbb{P}(A^t| \\Theta^t) = \\prod_{i<j} \\frac{e^{A^t_{ij}(\\theta_i^t + \\theta_j^t)}}{1 + e^{\\theta_i^t + \\theta_j^t}}`,

    with :math:`\\Theta^t \\equiv \\{\\theta_i^t\\}_{i = 1, \\dots, n}`.

    Parameters
    __________
    time_series: List object
        List of symmetric adjacency matrices [:math:`A_1, \\dots, A_T`] with binary entries and zero diagonal.
    tol: float
         Relative error of the estimated parameters. Default: 1e-2.
    maxit: integer
        Maximum number of iterations in the learning process. Default: 1e2.

    Returns
    _______

    phi_0: array
        Vector with the estimated values of the :math:`{\\varphi_0}_i`.
    phi_1: array
        Vector with the estimated values of :math:`{\\varphi_1}_i`,
    sigma: array
        Vector with the estimated values of :math:`\\sigma_i`,
    theta_naive: array_like
        Matrix that in the entry :math:`(i, t)` has a naive estimation of the :math:`\\theta_i^t`.
    theta: array_like
         Matrix that in the entry :math:`(i, t)` has the estimated values of :math:`\\theta_i^t`.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Create temporal network

    .. code:: python

     >>>    n = 60
     >>>    T = 150
     >>>    phi0 = np.ones(n) * 0.2
     >>>    phi1 = np.ones(n) * 0.6
     >>>    time_series = tgrg_simulation(n, T, phi_0=phi0, phi_1=phi1)

    Estimate the :math:`TGRG` model parameters

     .. code:: python

     >>>    phi_0, phi_1, sigma, theta_naive, theta = sm.tgrg(time_series)


    References
    __________

    .. [1] Mazzarisi, P., Barucca, P., Lillo, F. and Tantari, D., 2020.
           A dynamic network model with persistent links and node-specific latent variables, with an application to the interbank market.
           European Journal of Operational Research, 281(1), pp.50-65.
           https://doi.org/10.1016/j.ejor.2019.07.024



    """

    precision_0 = 1e-4
    precision_1 = 1e-4
    precision_phi = tol
    prec_learning = tol
    n = np.shape(time_series[0])[0]
    T = len(time_series)

    thetaESTNAIVE = np.zeros((n, T))

    for t in range(T):
        A_t = time_series[t]
        k = np.sum(A_t, 1)  # row sum
        ks = k.copy()
        ks[k == 0] = 1e-4
        # -------
        x_0 = np.random.rand(n, 1)
        x = np.zeros((n, 1))
        temp_pre = 1
        temp_ite = 1
        while temp_pre > precision_0 and temp_ite <= 10 * maxit:
            matrix_g = (np.ones((n, 1)) * np.reshape(x_0, (1, n))) / \
                       (1 + (np.reshape(x_0, (n, 1)) * np.reshape(x_0, (1, n))))
            matrix_g = matrix_g - np.diag(np.diag(matrix_g))
            x = ks / np.sum(matrix_g, 1)
            g = (x - x_0) / x_0
            g[k == 0] = 0
            temp_pre = np.max(abs(g)).copy()
            temp_ite = temp_ite + 1
            x_0 = x.copy()
        thetaESTNAIVE[:, t] = -np.log(x)
        if temp_ite > 10 * maxit:
            warnings.warn('Naive estimation: convergence at time %d not achieved' % t)
        # ------

    phi0_est = np.zeros((n, 1))
    phi1_est = np.zeros((n, 1))
    sigma_e = np.zeros((n, 1))

    for q in range(n):
        y = thetaESTNAIVE[q, :]
        model = ARIMA(y, order=[1, 0, 0], trend='c')
        fit = model.fit()
        model_params = fit.params
        phi0_est[q] = model_params[0]
        phi1_est[q] = model_params[1]
        sigma_e[q] = np.sqrt(model_params[2])

    phi0EST = phi0_est.copy()
    phi1EST = phi1_est.copy()
    sEST = sigma_e.copy()

    thetaEST0 = thetaESTNAIVE.copy()
    phi0E = phi0EST[:, -1].copy()
    phi1E = phi1EST[:, -1].copy()
    sE = sEST[:, -1].copy()
    precPHI = 1
    vecPrecPhi = []
    iterations = 0

    while (precPHI > precision_phi) and (iterations <= maxit):
        thetaEST1 = np.zeros((n, T))
        thetaEST1[:, 0] = thetaEST0[:, 0].copy()
        for t in range(1, T):
            A_t = time_series[t]
            k = np.sum(A_t, 1)
            theta0 = thetaEST0[:, t].copy()
            temp_pre = 1
            temp_ite = 1
            theta1 = np.zeros((n, 1))
            bound_0 = -30
            bound_1 = 30
            while (temp_pre > precision_1) and (temp_ite <= maxit):
                theta1 = np.zeros((n, 1))
                for i in range(n):
                    theta_bound_0 = grad_i(bound_0, i, k[i], theta0, phi0E[i], phi1E[i], sE[i],
                                           thetaEST0[i, t - 1])
                    theta_bound_1 = grad_i(bound_1, i, k[i], theta0, phi0E[i], phi1E[i], sE[i],
                                           thetaEST0[i, t - 1])
                    if theta_bound_0 * theta_bound_1 < 0:
                        theta1[i] = root_scalar(grad_i, args=(i, k[i], theta0, phi0E[i], phi1E[i], sE[i],
                                                              thetaEST0[i, t - 1]), bracket=(bound_0, bound_1)).root
                    else:
                        theta1[i] = thetaEST0[i, t - 1]
                temp_pre = np.max(abs(theta1 - theta0)).copy()
                theta0 = theta1.copy()
                temp_ite = temp_ite + 1
            thetaEST1[:, t] = np.reshape(theta1, n).copy()
        phi0_est = np.zeros((n, 1))
        phi1_est = np.zeros((n, 1))
        sigma_e = np.zeros((n, 1))
        for q in range(n):
            parameters = ipf_learning(q, T, time_series, thetaEST1, phi0E, phi1E, sE, prec_learning, maxit)
            phi0_est[q] = parameters[0].copy()
            phi1_est[q] = parameters[1].copy()
            sigma_e[q] = parameters[2].copy()

        phi0EST = np.append(phi0EST, phi0_est, axis=1)
        phi1EST = np.append(phi1EST, phi1_est, axis=1)
        sEST = np.append(sEST, sigma_e, axis=1)

        phi0_diff = phi0EST[:, -1] - phi0EST[:, -2]
        phi1_diff = phi1EST[:, -1] - phi1EST[:, -2]
        sEST_diff = sEST[:, -1] - sEST[:, -2]

        precPHI = max(np.nanmax(abs(phi0_diff)), np.nanmax(abs(phi1_diff)), np.nanmax(abs(sEST_diff)))
        vecPrecPhi = np.append(vecPrecPhi, precPHI)
        thetaEST0 = thetaEST1.copy()
        phi0E = phi0EST[:, -1].copy()
        phi1E = phi1EST[:, -1].copy()
        sE = sEST[:, -1].copy()

        if len(vecPrecPhi) >= 12 and (precPHI < 1) and (vecPrecPhi[iterations] > vecPrecPhi[iterations - 1]):
            precPHI = 0
        iterations = iterations + 1
    return -phi0EST[:, -1], phi1EST[:, -1], sEST[:, -1], -thetaESTNAIVE, -thetaEST0


def grad_i(theta, i, k, thetaV, phi0, phi1, sigma, thetaP):
    F = np.exp(-theta - thetaV) / (1 + np.exp(- theta - thetaV))
    F[i] = 0
    gi = -k + np.sum(F) - (theta - phi0 - phi1 * thetaP) / (sigma ** 2)

    return gi


def kernel_theta_0(x, y, i, vecA, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    f = np.zeros((1, L))
    for k in range(L):
        F = np.exp(-vecA * (x[k] + y)) / (1 + np.exp(-x[k] - y))
        F[i] = 1
        f[k] = np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return f


def kernel_theta_1(x, y, i, vecA, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    g1 = np.zeros((1, L))
    for k in range(L):
        F = np.exp(-vecA * (x[k] + y)) / (1 + np.exp(-x[k] - y))
        F[i] = 1
        g1[k] = x[k] * np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return g1


def kernel_theta_2(x, y, i, vecA, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    g2 = np.zeros((1, L))
    for k in range(L):
        F = np.exp(-vecA * (x[k] + y)) / (1 + np.exp(-x[k] - y))
        F[i] = 1
        g2[k] = (x[k] ** 2) * np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 /
                                                  (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return g2


def ipf_learning(q, T, tempA, thetaEST, phi0E, phi1E, sE, prec_LEARNING, ite_learning):
    BOUND = 30
    xhat = thetaEST[q, :].copy()
    yhat = thetaEST.copy()
    phi0 = np.zeros((3, 1))
    phi0[0] = phi0E[q].copy()
    phi0[1] = phi1E[q].copy()
    phi0[2] = sE[q].copy()
    phi1 = np.zeros((3, 1))
    temp_prec = 1
    iterations = 1
    while (temp_prec > prec_LEARNING) and (iterations <= ite_learning):
        zt = np.zeros((T - 1, 1))
        g1t = np.zeros((T - 1, 1))
        g2t = np.zeros((T - 1, 1))
        for t in range(1, T):
            y = yhat[:, t].copy()
            i = q
            vec_A = tempA[t][i, :].copy()
            zt[t - 1] = quad(kernel_theta_0, -BOUND, BOUND, args=(y, i, vec_A, phi0[0], phi0[1], phi0[2], xhat[t - 1]),
                             points=0)[0]
            g1t[t - 1] = quad(kernel_theta_1, -BOUND, BOUND, args=(y, i, vec_A, phi0[0], phi0[1], phi0[2], xhat[t - 1]),
                              points=0)[0]
            g2t[t - 1] = quad(kernel_theta_2, -BOUND, BOUND, args=(y, i, vec_A, phi0[0], phi0[1], phi0[2], xhat[t - 1]),
                              points=0)[0]

        tphi = np.mean(g1t / zt)
        t2phi = np.mean(g2t / zt)
        that = np.mean(xhat[:T - 1])
        t2hat = np.mean(xhat[:T - 1] ** 2)
        thattphi = np.mean(np.reshape(xhat[:T - 1], (1, T - 1)) * np.reshape(g1t / zt, (1, T - 1)))
        phi1[1] = (thattphi - that * tphi) / (t2hat - that ** 2)
        phi1[0] = tphi - that * phi1[1]
        phi1[2] = np.sqrt(abs(t2phi + phi1[0] ** 2 + t2hat * phi1[1] ** 2 - 2 * tphi * phi1[0] - 2 * thattphi * phi1[1]
                              + 2 * that * phi1[0] * phi1[1]))
        temp_prec = np.max(abs((phi1 - phi0) / phi0))
        phi0 = phi1.copy()
        iterations = iterations + 1
    parameters = phi1.copy()
    return parameters


def tgrg_simulation(n, T, theta_0=None, phi_0=None, phi_1=None, sigma=None):
    """
    Simulate a temporal network following the Temporally Generalized Random Graph model (:math:`TGRG`).

    For the :math:`TGRG` model [1]_, with temporal network represented by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}`, each node :math:`i` is described by a latent variable :math:`\\theta_i^t`, namely the fitness of the node, which evolves over time by following a covariance stationary autoregressive process :math:`AR(1)`:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\theta_i^t = \\varphi_{0,i} + \\varphi_{1,i} \\theta_i^{t-1} + \\epsilon_i^t`,

    with :math:`\\varphi_{0,i}\\in \\mathbb{R}`, :math:`|\\varphi_{1,i}|<1`, and i.i.d. normal innovations :math:`\\epsilon_i^t\\sim \\mathcal{N}(0, \\sigma_i^2)`.
    Then, the observation equation for the network snapshot at time :math:`t` is given by :math:`\\binom{N}{2}` independent Bernoulli trials whose conditional probability is:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\mathbb{P}(A^t| \\Theta^t) = \\prod_{i<j} \\frac{e^{A^t_{ij}(\\theta_i^t + \\theta_j^t)}}{1 + e^{\\theta_i^t + \\theta_j^t}}`,

    with :math:`\\Theta^t \\equiv \\{\\theta_i^t\\}_{i = 1, \\dots, n}`.

    Parameters
    __________

    n: integer
        Number of nodes in the graph.
    T: integer
         Number of time snapshots.
    theta_0: array
        Vector with the values of :math:`\\theta_i^0`. If it is set to 'None' the null vector is used. Default: 'None'.
    phi_0: array
        Vector with the values of :math:`{\\varphi_0}_i`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    phi_1: array
        Vector with the values of :math:`{\\varphi_1}_i`. If it is set to 'None' the vector of all 0.5 is used. Default: 'None'.
    sigma: array
        Vector with the values of :math:`{\\sigma}_i`. If it is set to 'None' the vector of all 0.3 is used . Default: 'None'.

    :return: **simulation**: (list)
        Temporal network produced by a :math:`TGRG` model.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Define input parameters

    .. code:: python

     >>>    n = 60
     >>>    T = 150
     >>>    phi_0 = np.ones(n) * 0.2
     >>>    phi_1 = np.ones(n) * 0.6

    Simulate the temporal network

     .. code:: python

     >>>    time_series = sm.tgrg_simulation(n, T, phi_0=phi_0, phi_1=phi_1)

     References
     __________

     .. [1] Mazzarisi, P., Barucca, P., Lillo, F. and Tantari, D., 2020.
       A dynamic network model with persistent links and node-specific latent variables, with an application to the interbank market.
       European Journal of Operational Research, 281(1), pp.50-65.
       https://doi.org/10.1016/j.ejor.2019.07.024



    """
    # generate latent variables
    if phi_0 is None:
        phi_0 = np.ones(n) * 0.3
    if phi_1 is None:
        phi_1 = np.ones(n) * 0.5
    if theta_0 is None:
        theta_0 = phi_0 / (1 - phi_1)
    if sigma is None:
        sigma = np.ones(n) * 0.3
    theta = np.zeros((n, T))
    theta[:, 0] = theta_0
    for i in range(n):
        for t in range(1, T):
            eps_i_t = np.random.normal(scale=sigma[i])
            theta[i, t] = phi_0[i] + theta[i, t - 1] * phi_1[i] + eps_i_t

    # generate adjacency matrices

    time_series = []
    for t in range(T):
        A_t = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                p_ij = np.exp(theta[i, t] + theta[j, t]) / (1 + np.exp(theta[i, t] + theta[j, t]))
                A_t[i, j] = np.random.choice([1, 0], 1, p=[p_ij, 1 - p_ij])
                A_t[j, i] = A_t[i, j]
        time_series.append(A_t)
    return time_series


def tgrg_directed(time_series, tol=1e-2, maxit=1e2):
    """
    Estimate, by an expectation-maximization algorithm, the parameters of the Temporally Generalized Random Graph model (:math:`TGRG`).

    In the :math:`TGRG` model for directed graphs [1]_ , with temporal network represented by a time series of non-symmetric adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}`, each node :math:`i` is characterized by two latent variables, :math:`\\theta_i^{t,in}` and :math:`\\theta_i^{t,out}`, both of them evolving in time by following a covariance stationary autoregressive process :math:`AR(1)`:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa} \\theta_i^{t, in} = \\varphi_{0,i}^{in} + \\varphi_{1,i}^{in} \\theta_i^{t-1, in} + \\epsilon_i^{t, in}`,

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa} \\theta_i^{t, out} = \\varphi_{0,i}^{out} + \\varphi_{1,i}^{out} \\theta_i^{t-1, out} + \\epsilon_i^{t, out},`

    where :math:`\\varphi_{0,i}^{in}, \\varphi_{0,i}^{out}\\in \\mathbb{R}`, :math:`|\\varphi_{1,i}^{in}|, |\\varphi_{1,i}^{out}|<1`, and i.i.d. normal innovations :math:`\\epsilon_i^{t, in}\\sim \\mathcal{N}(0, {\\sigma_i^{in}}^{2})` and :math:`\\epsilon_i^{t, out}\\sim \\mathcal{N}(0, {\\sigma_i^{out}}^2)`.

    Then, the observation equation for the network snapshot at time :math:`t` is given by :math:`N(N-1)` independent Bernoulli trials whose conditional probability is:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\mathbb{P}(A^t| \\Theta^t) = \\prod_{j\\neq i} \\frac{e^{A^t_{ij}(\\theta_i^{t, out} + \\theta_j^{t, in})}}{1 + e^{\\theta_i^{t, out} + \\theta_j^{t, in}}},`

    where :math:`\\Theta^t \\equiv \\{\\theta_i^{t, in}, \\theta_i^{t, out}\\}_{i = 1, \\dots, n}`.

    Parameters
    __________
    time_series: List object
        List of adjacency matrices [:math:`A_1, \\dots, A_T`] with binary entries and zero diagonal.
    tol: float
         Relative error of the estimated parameters. Default: 1e-2.
    maxit: integer
        Maximum number of iterations in the learning process. Default: 1e2.

    Returns
    _______

    phi_0: array
        Vector of length :math:`2n` with the estimated values of :math:`\\varphi_0`. The first :math:`n` entries contain :math:`\\varphi_0^{out}`, while the last :math:`n` entries contain :math:`\\varphi_0^{in}`.
    phi_1: array
        Vector of length :math:`2n` with the estimated values of :math:`\\varphi_1`. The first :math:`n` entries contain :math:`\\varphi_1^{out}`, while the last :math:`n` entries contain :math:`\\varphi_1^{in}`.
    sigma: array
        Vector of length :math:`2n` with the estimated values of :math:`\\sigma`. The first :math:`n` entries contain :math:`\\sigma^{out}`, while the last :math:`n` entries contain :math:`\\sigma^{in}`.
    theta_naive: array_like
        Matrix of size :math:`2n \\times T`, that in the entry :math:`(i, t)` has a naive estimation of the :math:`\\theta` parameters. The first :math:`n` rows contain the values of the :math:`{\\theta^{t}_i}^{out}` while the last :math:`n` rows contain the :math:`{\\theta^{t}_i}^{in}`.
    theta: array_like
        Matrix of size :math:`2n \\times T`, that in the entry :math:`(i, t)` has the estimation of the :math:`\\theta` parameters. The first :math:`n` rows contain the values of the :math:`{\\theta^{t}_i}^{out}` while the last :math:`n` rows contain the :math:`{\\theta^{t}_i}^{in}`.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Create temporal network

    .. code:: python

     >>>    n = 60
     >>>    T = 150
     >>>    phi0 = np.ones(n) * 0.2
     >>>    phi1 = np.ones(n) * 0.6
     >>>    time_series = tgrg_directed_simulation(n, T, phi_0_in=phi0, phi_0_out=phi0, phi_1_in=phi1, phi_1_out=phi1)

    Estimate the :math:`TGRG` model parameters

     .. code:: python

     >>>    phi_0, phi_1, sigma, theta_naive, theta = sm.tgrg_directed(time_series)


    References
    __________

    .. [1] Mazzarisi, P., Barucca, P., Lillo, F. and Tantari, D., 2020.
       A dynamic network model with persistent links and node-specific latent variables, with an application to the interbank market.
       European Journal of Operational Research, 281(1), pp.50-65.
       https://doi.org/10.1016/j.ejor.2019.07.024

    """

    n = np.shape(time_series[0])[0]
    T = len(time_series)
    precision_0 = tol
    precision_1 = tol
    precision_phi = tol
    prec_learning = tol

    thetaESTNAIVE = np.zeros((2 * n, T))

    for t in range(T):
        A_t = time_series[t]
        k_out = np.sum(A_t, 1)
        k_in = np.sum(A_t, 0)
        k = np.append(np.reshape(k_out, (n, 1)), np.reshape(k_in, (n, 1)))
        ks = k.copy()
        ks[k == 0] = 1e-4
        x_0 = np.random.rand(2 * n, 1)
        x = np.zeros((2 * n, 1))
        temp_pre = 1
        temp_ite = 1
        while temp_pre > precision_0 and temp_ite <= maxit:
            xfOut = x_0[:n].copy()
            xfIn = x_0[n:].copy()
            matrix_gi = (np.ones((n, 1)) * np.reshape(xfIn, (1, n))) / \
                        (1 + (np.reshape(xfOut, (n, 1)) * np.reshape(xfIn, (1, n))))
            matrix_gj = (np.ones((n, 1)) * np.reshape(xfOut, (1, n))) / \
                        (1 + (np.reshape(xfIn, (n, 1)) * np.reshape(xfOut, (1, n))))
            matrix_gi = matrix_gi - np.diag(np.diag(matrix_gi))
            matrix_gj = matrix_gj - np.diag(np.diag(matrix_gj))
            matrix_g = np.zeros((2 * n, n))
            matrix_g[:n, :] = matrix_gi.copy()
            matrix_g[n:, :] = matrix_gj.copy()
            x = ks / np.sum(matrix_g, 1)
            g = (x - x_0) / x_0
            g[k == 0] = 0
            temp_pre = np.max(abs(g)).copy()
            temp_ite = temp_ite + 1
            x_0 = x.copy()
        x_out = x[:n].copy()
        x_in = x[n:].copy()
        x_out = x_out / x_out[0]
        x_in = x_in * x_out[0]
        x = np.append(np.reshape(x_out, (n, 1)), np.reshape(x_in, (n, 1)))
        thetaESTNAIVE[:, t] = -np.log(x)
        if temp_ite > maxit:
            warnings.warn('Naive estimation: convergence at time %d not achieved' % t)
        # -----------------------

    phi0_est = np.zeros((2 * n, 1))
    phi1_est = np.zeros((2 * n, 1))
    sigma_e = np.zeros((2 * n, 1))

    for q in range(1, 2 * n):
        y = thetaESTNAIVE[q, :]
        model = ARIMA(y, order=[1, 0, 0], trend='c')
        fit = model.fit()
        model_params = fit.params
        phi0_est[q] = model_params[0]
        phi1_est[q] = model_params[1]
        sigma_e[q] = np.sqrt(model_params[2])

    phi0EST = phi0_est.copy()
    phi1EST = phi1_est.copy()
    sEST = sigma_e.copy()

    thetaEST0 = thetaESTNAIVE.copy()
    phi0E = phi0EST[:, -1].copy()
    phi1E = phi1EST[:, -1].copy()
    sE = sEST[:, -1].copy()
    precPHI = 1
    vecPrecPhi = []
    iterations = 0

    while (precPHI > precision_phi) and (iterations <= maxit):
        thetaEST1 = np.zeros((2 * n, T))
        thetaEST1[:, 0] = thetaEST0[:, 0].copy()
        for t in range(1, T):
            A_t = time_series[t].copy()
            k_out = np.sum(A_t, 1)
            k_in = np.sum(A_t, 0)
            k = np.append(np.reshape(k_out, (n, 1)), np.reshape(k_in, (n, 1)))
            theta0 = thetaEST0[:, t].copy()
            temp_pre = 1
            temp_ite = 1
            theta1 = np.zeros((2 * n, 1))
            while (temp_pre > precision_1) and (temp_ite <= maxit):
                theta0OUT = theta0[:n].copy()
                theta0IN = theta0[n:].copy()
                theta1 = np.zeros((2 * n, 1))
                bound_0 = -30
                bound_1 = 30
                for i in range(1, 2 * n):
                    if i < n:
                        theta1[i] = root_scalar(grad_i_directed, args=(i, k[i], theta0IN, phi0E[i], phi1E[i], sE[i],
                                                                       thetaEST0[i, t - 1]),
                                                bracket=(bound_0, bound_1)).root
                    else:
                        theta1[i] = root_scalar(grad_i_directed,
                                                args=(i - n, k[i], theta0OUT, phi0E[i], phi1E[i], sE[i],
                                                      thetaEST0[i, t - 1]), bracket=(bound_0, bound_1)).root
                temp_pre = np.max(abs(theta1 - theta0)).copy()
                theta0 = theta1.copy()
                temp_ite = temp_ite + 1
            thetaEST1[:, t] = np.reshape(theta1, 2 * n).copy()
        phi0_est = np.zeros((2 * n, 1))
        phi1_est = np.zeros((2 * n, 1))
        sigma_e = np.zeros((2 * n, 1))
        for q in range(1, 2 * n):
            parameters = ipf_learning_directed(q, n, T, time_series, thetaEST1, phi0E, phi1E, sE, prec_learning, maxit)
            phi0_est[q] = parameters[0].copy()
            phi1_est[q] = parameters[1].copy()
            sigma_e[q] = parameters[2].copy()

        phi0EST = np.append(phi0EST, phi0_est, axis=1)
        phi1EST = np.append(phi1EST, phi1_est, axis=1)
        sEST = np.append(sEST, sigma_e, axis=1)

        phi0_diff = (phi0EST[1:, -1] - phi0EST[1:, -2]) / phi0EST[1:, -2]
        phi1_diff = (phi1EST[1:, -1] - phi1EST[1:, -2]) / phi1EST[1:, -2]
        sEST_diff = (sEST[1:, -1] - sEST[1:, -2]) / sEST[1:, -2]

        precPHI = max(np.nanmax(abs(phi0_diff)), np.nanmax(abs(phi1_diff)), np.nanmax(abs(sEST_diff)))

        vecPrecPhi = np.append(vecPrecPhi, precPHI)
        thetaEST0 = thetaEST1.copy()
        phi0E = phi0EST[:, -1].copy()
        phi1E = phi1EST[:, -1].copy()
        sE = sEST[:, -1].copy()

        if len(vecPrecPhi) >= 12 and (precPHI < 1) and (vecPrecPhi[iterations] > vecPrecPhi[iterations - 1]):
            precPHI = 0
        iterations = iterations + 1
    return -phi0EST[:, -1], phi1EST[:, -1], sEST[:, -1], -thetaESTNAIVE, -thetaEST0


def kernel_theta_0_directed(x, y, i, vecA, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    f = np.zeros((1, L))
    for k in range(L):
        F = np.exp(-vecA * (x[k] + y)) / (1 + np.exp(-x[k] - y))
        F[i] = 1
        f[k] = np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return f


def kernel_theta_1_directed(x, y, i, vecA, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    g1 = np.zeros((1, L))
    for k in range(L):
        F = np.exp(-vecA * (x[k] + y)) / (1 + np.exp(-x[k] - y))
        F[i] = 1
        g1[k] = x[k] * np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return g1


def kernel_theta_2_directed(x, y, i, vecA, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    g2 = np.zeros((1, L))
    for k in range(L):
        F = np.exp(-vecA * (x[k] + y)) / (1 + np.exp(-x[k] - y))
        F[i] = 1
        g2[k] = (x[k] ** 2) * np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 /
                                                  (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return g2


def grad_i_directed(theta, i, k, thetaV, phi0, phi1, sigma, thetaP):
    F = np.exp(-theta - thetaV) / (1 + np.exp(- theta - thetaV))
    F[i] = 0
    gi = -k + np.sum(F) - (theta - phi0 - phi1 * thetaP) / (sigma ** 2)

    return gi


def ipf_learning_directed(q, n, T, tempA, thetaEST, phi0E, phi1E, sE, prec_LEARNING, ite_learning):
    BOUND = 30
    if q < n:
        xhat = thetaEST[q, :].copy()
        yhat = thetaEST[n:, :].copy()
    else:
        xhat = thetaEST[q, :].copy()
        yhat = thetaEST[:n, :].copy()
    phi0 = np.zeros((3, 1))
    phi0[0] = phi0E[q].copy()
    phi0[1] = phi1E[q].copy()
    phi0[2] = sE[q].copy()
    phi1 = np.zeros((3, 1))
    temp_prec = 1
    iterations = 0

    while (temp_prec > prec_LEARNING) and (iterations <= ite_learning):
        zt = np.zeros((T - 1, 1))
        g1t = np.zeros((T - 1, 1))
        g2t = np.zeros((T - 1, 1))
        for t in range(1, T):
            y = yhat[:, t].copy()
            if q < n:
                i = q
                vec_A = tempA[t][i, :].copy()
            else:
                i = q - n
                vec_A = tempA[t][:, i].copy()
            zt[t - 1] = quad(kernel_theta_0_directed, -BOUND, BOUND, args=(y, i, vec_A, phi0[0], phi0[1], phi0[2],
                                                                           xhat[t - 1]), points=0)[0]
            g1t[t - 1] = quad(kernel_theta_1_directed, -BOUND, BOUND, args=(y, i, vec_A, phi0[0], phi0[1], phi0[2],
                                                                            xhat[t - 1]), points=0)[0]
            g2t[t - 1] = quad(kernel_theta_2_directed, -BOUND, BOUND, args=(y, i, vec_A, phi0[0], phi0[1], phi0[2],
                                                                            xhat[t - 1]), points=0)[0]
        tphi = np.mean(g1t / zt)
        t2phi = np.mean(g2t / zt)
        that = np.mean(xhat[:T - 1])
        t2hat = np.mean(xhat[:T - 1] ** 2)
        thattphi = np.mean(np.reshape(xhat[:T - 1], (1, T - 1)) * np.reshape(g1t / zt, (1, T - 1)))
        phi1[1] = (thattphi - that * tphi) / (t2hat - that ** 2)
        phi1[0] = tphi - that * phi1[1]
        phi1[2] = np.sqrt(abs(t2phi + phi1[0] ** 2 + t2hat * phi1[1] ** 2 - 2 * tphi * phi1[0] - 2 * thattphi * phi1[1]
                              + 2 * that * phi1[0] * phi1[1]))
        temp_prec = np.max(abs((phi1 - phi0) / phi0))
        phi0 = phi1.copy()
        iterations = iterations + 1
    parameters = phi1.copy()
    return parameters


def tgrg_directed_simulation(n, T, theta_0_in=None, theta_0_out=None,
                             phi_0_in=None, phi_0_out=None,
                             phi_1_in=None, phi_1_out=None,
                             sigma_in=None, sigma_out=None):
    """
    Simulate a *directed* temporal network following the Temporally Generalized Random Graph model (:math:`TGRG`).

    In the :math:`TGRG` model for directed graphs [1]_, with temporal network described by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}` with :math:`A_{ij}^t\\neq A_{ji}^t` in general,

    each node :math:`i` is characterized by two latent variables, :math:`\\theta_i^{t,in}` and :math:`\\theta_i^{t,out}`, both of them evolving in time by following a covariance stationary autoregressive process :math:`AR(1)`:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa} \\theta_i^{t, in} = \\varphi_{0,i}^{in} + \\varphi_{1,i}^{in} \\theta_i^{t-1, in} + \\epsilon_i^{t, in}`,

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa} \\theta_i^{t, out} = \\varphi_{0,i}^{out} + \\varphi_{1,i}^{out} \\theta_i^{t-1, out} + \\epsilon_i^{t, out},`

    where :math:`\\varphi_{0,i}^{in}, \\varphi_{0,i}^{out}\\in \\mathbb{R}`, :math:`|\\varphi_{1,i}^{in}|, |\\varphi_{1,i}^{out}|<1`, and i.i.d. normal innovations :math:`\\epsilon_i^{t, in}\\sim \\mathcal{N}(0, {\\sigma_i^{in}}^{2})` and :math:`\\epsilon_i^{t, out}\\sim \\mathcal{N}(0, {\\sigma_i^{out}}^2)`.

    Then, the observation equation for the network snapshot at time :math:`t` is given by :math:`N(N-1)` independent Bernoulli trials whose conditional probability is:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\mathbb{P}(A^t| \\Theta^t) = \\prod_{j\\neq i} \\frac{e^{A^t_{ij}(\\theta_i^{t, out} + \\theta_j^{t, in})}}{1 + e^{\\theta_i^{t, out} + \\theta_j^{t, in}}},`

    where :math:`\\Theta^t \\equiv \\{\\theta_i^{t, in}, \\theta_i^{t, out}\\}_{i = 1, \\dots, n}`.

    Parameters
    __________

    n: integer
        Number of nodes in the graph.
    T: integer
        Number of time snapshots.
    theta_0_in: array
        Vector with the values of :math:`{\\theta_i^0}^{in}`. If it is set to 'None' the null vector is used. Default: 'None'.
    theta_0_out: array
        Vector with the values of :math:`{\\theta_i^0}^{out}`. If it is set to 'None' the null vector is used. Default: 'None'.
    phi_0_in: array
        Vector with the values of :math:`{{\\varphi_0}_i}^{in}`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    phi_0_out: array
        Vector with the values of :math:`{{\\varphi_0}_i}^{out}`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    phi_1_in: array
        Vector with the values of :math:`{{\\varphi_1}_i}^{in}`. If it is set to 'None' the vector of all 0.5 is used. Default: 'None'.
    phi_1_out: array
        Vector with the values of :math:`{{\\varphi_1}_i}^{out}`. If it is set to 'None' the vector of all 0.5 is used. Default: 'None'.
    sigma_in: array
        Vector with the values of :math:`{{\\sigma}_i}^{in}`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    sigma_out: array
        Vector with the values of :math:`{{\\sigma}_i}^{out}`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.



    :return: **simulation**: (list)
        Directed temporal network produced by a :math:`TGRG` model.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Define input parameters

    .. code:: python

     >>>    n = 50
     >>>    T = 100

    Simulate the temporal network

     .. code:: python

    >>>   time_series = sm.tgrg_directed_simulation(n, T)

    References
    __________

    .. [1] Mazzarisi, P., Barucca, P., Lillo, F. and Tantari, D., 2020.
       A dynamic network model with persistent links and node-specific latent variables, with an application to the interbank market.
       European Journal of Operational Research, 281(1), pp.50-65.
       https://doi.org/10.1016/j.ejor.2019.07.024
    """
    # generate latent variables
    if phi_0_in is None:
        phi_0_in = np.ones(n) * 0.3
    if phi_0_out is None:
        phi_0_out = np.ones(n) * 0.3
    if phi_1_in is None:
        phi_1_in = np.ones(n) * 0.5
    if phi_1_out is None:
        phi_1_out = np.ones(n) * 0.5
    if theta_0_in is None:
        theta_0_in = phi_0_in / (1 - phi_1_in)
    if theta_0_out is None:
        theta_0_out = phi_0_out / (1 - phi_1_out)
    if sigma_in is None:
        sigma_in = np.ones(n) * 0.3
    if sigma_out is None:
        sigma_out = np.ones(n) * 0.3

    phi_0 = np.zeros(2 * n)
    phi_0[:n] = phi_0_out.copy()
    phi_0[n:] = phi_0_in.copy()

    phi_1 = np.zeros(2 * n)
    phi_1[:n] = phi_1_out.copy()
    phi_1[n:] = phi_1_in.copy()

    sigma = np.zeros(2 * n)
    sigma[:n] = sigma_out.copy()
    sigma[n:] = sigma_in.copy()

    theta = np.zeros((2 * n, T))
    theta[:n, 0] = theta_0_out.copy()
    theta[n:, 0] = theta_0_in.copy()

    for i in range(2 * n):
        for t in range(1, T):
            eps_i_t = np.random.normal(scale=sigma[i])
            theta[i, t] = phi_0[i] + theta[i, t - 1] * phi_1[i] + eps_i_t

    # generate adjacency matrices

    time_series = []
    for t in range(T):
        A_t = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j != i:
                    p_ij = np.exp(theta[i, t] + theta[j + n, t]) / (1 + np.exp(theta[i, t] + theta[j + n, t]))
                    A_t[i, j] = np.random.choice([1, 0], 1, p=[p_ij, 1 - p_ij])
        time_series.append(A_t)
    return time_series


def dar1_x_starting_point(A):
    T = len(A)
    bound0 = 0.001
    bound1 = 0.999
    if np.sum(A) == 0:
        chi = 0
        rho = 0
    elif np.sum(A) == T:
        chi = 1
        rho = 0
    else:
        Aplus = A[1:].copy()
        Aminus = A[:-1].copy()
        chi0 = np.sum(A) / T
        r0 = __likelihood_d_q(bound0, chi0, Aplus, Aminus)
        r1 = __likelihood_d_q(bound1, chi0, Aplus, Aminus)
        if r0 * r1 >= 0:
            chi = chi0
            rho = 0
        else:
            rho0 = root_scalar(__likelihood_d_q, args=(chi0, Aplus, Aminus), bracket=(bound0, bound1)).root
            chi = chi0
            rho = rho0

    return chi, rho


def grad_x_i(theta, i, At, At1, thetaV, alpha, phi0, phi1, sigma, thetaP):
    F = np.exp(-theta - thetaV) / (1 + np.exp(- theta - thetaV))
    IAi = np.array(1 * (At[i, :] == At1[i, :]))
    IAi[i] = 0
    Ait = At[i, :]
    ai = alpha[i, :]
    PAi = np.exp(-Ait * (theta + thetaV)) / (1 + np.exp(-theta - thetaV))
    g = (((1 - ai) * PAi) / (ai * IAi + (1 - ai) * PAi)) * (-Ait + F)
    g[i] = 0
    gi = np.sum(g) - (theta - phi0 - phi1 * thetaP) / (sigma ** 2)
    return gi


def kernel_theta_0_x(x, y, i, Ati, IAti, ai, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    f = np.zeros((1, L))
    for k in range(L):
        F = ai * IAti + (1 - ai) * (np.exp(-Ati * (x[k] + y)) / (1 + np.exp(-x[k] - y)))
        F[i] = 1
        f[k] = np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return f


def kernel_theta_1_x(x, y, i, Ati, IAti, ai, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    g1 = np.zeros((1, L))
    for k in range(L):
        F = ai * IAti + (1 - ai) * (np.exp(-Ati * (x[k] + y)) / (1 + np.exp(-x[k] - y)))
        F[i] = 1
        g1[k] = x[k] * np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return g1


def kernel_theta_2_x(x, y, i, Ati, IAti, ai, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    g2 = np.zeros((1, L))
    for k in range(L):
        F = ai * IAti + (1 - ai) * (np.exp(-Ati * (x[k] + y)) / (1 + np.exp(-x[k] - y)))
        F[i] = 1
        g2[k] = (x[k] ** 2) * np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (
                np.sqrt(2 * np.pi) * s)
    return g2


def ipf_learning_x(q, n, T, A, thetaE, alphaE, phi0E, phi1E, sE, prec_LEARNING, ite_learning):
    BOUND = 30
    xhat = thetaE[q, :].copy()
    yhat = thetaE.copy()
    phi0 = np.zeros((3, 1))
    phi0[0] = phi0E[q].copy()
    phi0[1] = phi1E[q].copy()
    phi0[2] = sE[q].copy()
    phi1 = np.zeros((3, 1))
    temp_prec = 1
    iterations = 0
    while (temp_prec > prec_LEARNING) and (iterations <= ite_learning):
        zt = np.zeros((T - 1, 1))
        g1t = np.zeros((T - 1, 1))
        g2t = np.zeros((T - 1, 1))
        for t in range(1, T):
            y = yhat[:, t].copy()
            i = q
            Ati = A[t][i, :]
            IAti = 1 * (A[t][i, :] == A[t - 1][i, :])
            ai = alphaE[i, :]
            zt[t - 1] = quad(kernel_theta_0_x, -BOUND, BOUND, args=(y, i, Ati, IAti, ai, phi0[0], phi0[1], phi0[2],
                                                                    xhat[t - 1]), points=0)[0]
            g1t[t - 1] = quad(kernel_theta_1_x, -BOUND, BOUND, args=(y, i, Ati, IAti, ai, phi0[0], phi0[1], phi0[2],
                                                                     xhat[t - 1]), points=0)[0]
            g2t[t - 1] = quad(kernel_theta_2_x, -BOUND, BOUND, args=(y, i, Ati, IAti, ai, phi0[0], phi0[1], phi0[2],
                                                                     xhat[t - 1]), points=0)[0]
        tphi = np.mean(g1t / zt)
        t2phi = np.mean(g2t / zt)
        that = np.mean(xhat[:T - 1])
        t2hat = np.mean(xhat[:T - 1] ** 2)
        thattphi = np.mean(np.reshape(xhat[:T - 1], (1, T - 1)) * np.reshape(g1t / zt, (1, T - 1)))
        phi1[1] = (thattphi - that * tphi) / (t2hat - that ** 2)
        phi1[0] = tphi - that * phi1[1]
        phi1[2] = np.sqrt(abs(t2phi + phi1[0] ** 2 + t2hat * phi1[1] ** 2 - 2 * tphi * phi1[0] - 2 * thattphi * phi1[1]
                              + 2 * that * phi1[0] * phi1[1]))
        temp_prec = np.max(abs((phi1 - phi0) / phi0))
        phi0 = phi1.copy()
        iterations = iterations + 1
    parameters = phi1.copy()
    return parameters


def auxiliary_alpha_x(a, IA, KernelA):
    s = np.sum(IA / (a * IA + (1 - a) * KernelA) - KernelA / (a * IA + (1 - a) * KernelA))
    return s


def ipf_learning_alpha_x(At, T, i, j, thetai, thetaj, phi0xi, phi1xi, sexi, phi0xj, phi1xj, sexj, pKw, xKw):
    bound0 = 1e-3
    bound1 = 0.999
    Apost = At[1:]
    Apre = At[:-1]
    IA = 1 * (Apost == Apre)
    KernelA = np.zeros((T - 1, 1))
    for t in range(1, T):
        KernelA[t - 1] = np.trapz((1 / 2) * pKw * (1 / np.sqrt(1 + xKw * (sexi ** 2 + sexj ** 2)))
                                  * np.exp(((1 - 2 * At[t]) ** 2 * (sexi ** 2 + sexj ** 2) +
                                            4 * (phi0xi + phi1xi * thetai[t - 1] + phi0xj + phi1xj * thetaj[t - 1])
                                            * (1 - 2 * At[t] - xKw *
                                               (phi0xi + phi1xi * thetai[t - 1] + phi0xj + phi1xj * thetaj[t - 1]))) /
                                           (8 * (1 + xKw * (sexi ** 2 + sexj ** 2)))), xKw)

    ga0 = np.sum(IA / (bound0 * IA + (1 - bound0) * KernelA) - KernelA / (bound0 * IA + (1 - bound0) * KernelA))
    ga1 = np.sum(IA / (bound1 * IA + (1 - bound1) * KernelA) - KernelA / (bound1 * IA + (1 - bound1) * KernelA))

    if ga0 * ga1 < 0:
        alphaij = root_scalar(auxiliary_alpha_x, args=(IA, KernelA), bracket=(bound0, bound1)).root
    else:
        alphaij = 0
    return alphaij


def dar_tgrg(time_series, tol=1e-2, maxit=1e2):
    """
    Estimate, by an expectation-maximization algorithm, the parameters of the Discrete Auto-Regressive Temporally Generalized Random Graph model (:math:`DAR`-:math:`TGRG`).

    The :math:`DAR`-:math:`TGRG` model [1]_ can be interpreted as a mixture of :math:`DAR` and :math:`TGRG` models where the persistence pattern associated with the copying mechanism of the :math:`DAR` model coexists with the
    node fitnesses evolving in time according to the :math:`TGRG` model.

    In the :math:`DAR`-:math:`TGRG` model, with temporal network described by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}` each node is characterized by a latent variable :math:`\\theta_i^t`, namely the fitness of the node, which evolves in time by following a covariance stationary autoregressive process :math:`AR(1)`:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\theta_i^t = \\varphi_{0,i} + \\varphi_{1,i} \\theta_i^{t-1} + \\epsilon_i^t,`

    with :math:`\\varphi_{0,i}\\in \\mathbb{R}`, :math:`|\\varphi_{1,i}|<1`, and i.i.d. normal innovations :math:`\\epsilon_i^t\\sim \\mathcal{N}(0, \\sigma_i^2)`.

    Then, the observation equation for the network snapshot at time :math:`t` is given by :math:`\\binom{N}{2}` independent Bernoulli trials whose conditional probability is:

    :math:`\\phantom{aaaaaaaa}\\mathbb{P}(A^t| \\Theta^t, A^{t-1}, \\mathbf{\\alpha}) = \\prod_{i<j}\\left( \\alpha_{ij}\\mathbb{I}_{A^t_{ij}A^{t-1}_{ij}} + (1-\\alpha_{ij}) \\frac{e^{A^t_{ij}(\\theta_i^t + \\theta_j^t)}}{1 + e^{\\theta_i^t + \\theta_j^t}}\\right),`

    where :math:`\\Theta^t \\equiv \\{\\theta_i^t\\}_{i = 1, \\dots, n}` and :math:`\\alpha \\equiv  \\{\\alpha_{ij}\\}_{i,j = 1, \\dots, n}` with :math:`0<\\alpha_{ij}<1`.

    Parameters
    __________
    time_series: list
        List of symmetric adjacency matrices [:math:`A_1, \\dots, A_T`].
    tol: float
        Relative error of the estimated parameters. Default: 1e-2.
    maxit: integer
        Maximum number of iterations in the learning process. Default: 1e2.

    Returns
    _______

    phi_0: array
        Vector with the estimated values of the :math:`{\\varphi_0}_i`.
    phi_1: array
        Vector with the estimated values of :math:`{\\varphi_1}_i`,
    sigma: array
        Vector with the estimated values of :math:`\\sigma_i`,
    theta_naive: array_like
        Matrix that in the entry :math:`(i, t)` has a naive estimation of the :math:`\\theta_i^t`.
    theta: array_like
         Matrix that in the entry :math:`(i, t)` has the estimated values of :math:`\\theta_i^t`.
    alpha: array_like
        Matrix with the estimated values of :math:`\\alpha_{ij}`.


    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Create temporal network

    .. code:: python

     >>>    n = 60
     >>>    T = 150
     >>>    phi0 = np.ones(n) * 0.2
     >>>    time_series = dar_tgrg_simulation(n, T, phi_0=phi0)

    Estimate the :math:`DAR`-:math:`TGRG` model parameters

    .. code:: python

     >>>    phi_0, phi_1, sigma, theta_naive, theta, alpha = sm.dar_tgrg(time_series)

    References
    __________

    .. [1] Mazzarisi, P., Barucca, P., Lillo, F. and Tantari, D., 2020.
       A dynamic network model with persistent links and node-specific latent variables, with an application to the interbank market.
       European Journal of Operational Research, 281(1), pp.50-65.
       https://doi.org/10.1016/j.ejor.2019.07.024
    """

    with open('polya_gamma_points', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    xKw = np.zeros(len(data))
    for i in range(len(data)):
        xKw[i] = float(data[i][1])

    with open('polya_gamma_values', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    pKw = np.zeros(len(data))
    for i in range(len(data)):
        pKw[i] = float(data[i][1])

    precision_0 = 1e-4
    precision_1 = 1e-4
    precision_phi = tol
    prec_learning = tol
    n = np.shape(time_series[0])[0]
    T = len(time_series)

    thetaESTNAIVEx = np.zeros((n, T))

    for t in range(T):
        A_t = time_series[t]
        k = np.sum(A_t, 1)  # row sum
        ks = k.copy()
        ks[k == 0] = 1e-4
        # -------
        x_0 = np.random.rand(n, 1)
        x = np.zeros((n, 1))
        temp_pre = 1
        temp_ite = 1
        while temp_pre > precision_0 and temp_ite <= 10 * maxit:
            matrix_g = (np.ones((n, 1)) * np.reshape(x_0, (1, n))) / \
                       (1 + (np.reshape(x_0, (n, 1)) * np.reshape(x_0, (1, n))))
            matrix_g = matrix_g - np.diag(np.diag(matrix_g))
            x = ks / np.sum(matrix_g, 1)
            g = (x - x_0) / x_0
            g[k == 0] = 0
            temp_pre = np.max(abs(g)).copy()
            temp_ite = temp_ite + 1
            x_0 = x.copy()
        thetaESTNAIVEx[:, t] = -np.log(x)
        if temp_ite > 10 * maxit:
            warnings.warn('Naive estimation: convergence at time %d not achieved' % t)

    phi0_x = np.zeros((n, 1))
    phi1_x = np.zeros((n, 1))
    s_ex = np.zeros((n, 1))

    for q in range(n):
        y = thetaESTNAIVEx[q, :]
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            model = ARIMA(y, order=[1, 0, 0], trend='c')
            try:
                fit = model.fit()
            except Warning:
                warnings.filterwarnings('default')
                warnings.warn('Naive estimation: convergence at time %d not achieved.' % q)
        model_params = fit.params
        phi0_x[q] = model_params[0]
        phi1_x[q] = model_params[1]
        s_ex[q] = np.sqrt(model_params[2])

    phi0ESTx = phi0_x.copy()
    phi1ESTx = phi1_x.copy()
    sESTx = s_ex.copy()

    alpha0NAIVEx = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vecA = np.zeros(T)
            for t in range(T):
                vecA[t] = time_series[t][i, j]
            chiest, rhoest = dar1_x_starting_point(vecA)
            alpha0NAIVEx[i, j] = rhoest

    alpha0NAIVEx = (alpha0NAIVEx + alpha0NAIVEx.transpose()).copy()
    thetaEST0x = thetaESTNAIVEx.copy()
    phi0Ex = phi0ESTx[:, -1]
    phi1Ex = phi1ESTx[:, -1]
    sEx = sESTx[:, -1]
    alpha0x = alpha0NAIVEx.copy()
    precPHI = 1
    vecPrecPhi = []
    iterations = 0

    while (precPHI > precision_phi) and (iterations <= maxit):
        thetaEST1x = np.zeros((n, T))
        thetaEST1x[:, 0] = thetaEST0x[:, 0].copy()
        for t in range(1, T):
            At = time_series[t].copy()
            At1 = time_series[t - 1].copy()
            theta0x = thetaEST0x[:, t].copy()
            temp_pre = 1
            temp_ite = 1
            bound_0 = -30
            bound_1 = 30
            theta1x = np.zeros((n, 1))
            while (temp_pre > precision_1) and (temp_ite <= maxit):
                theta1x = np.zeros((n, 1))
                for i in range(n):
                    theta_bound_0 = grad_x_i(bound_0, i, At, At1, theta0x, alpha0x, phi0Ex[i], phi1Ex[i],
                                             sEx[i], thetaEST0x[i, t - 1])
                    theta_bound_1 = grad_x_i(bound_1, i, At, At1, theta0x, alpha0x, phi0Ex[i], phi1Ex[i],
                                             sEx[i], thetaEST0x[i, t - 1])
                    if theta_bound_0 * theta_bound_1 < 0:
                        theta1x[i] = root_scalar(grad_x_i, args=(i, At, At1, theta0x, alpha0x, phi0Ex[i], phi1Ex[i],
                                                                 sEx[i], thetaEST0x[i, t - 1]),
                                                 bracket=(bound_0, bound_1)).root
                    else:
                        theta1x[i] = thetaEST0x[i, t - 1]

                temp_pre = np.max(abs(theta1x - theta0x)).copy()
                theta0x = theta1x.copy()
                temp_ite = temp_ite + 1
            thetaEST1x[:, t] = np.reshape(theta1x, n).copy()

        phi0_x = np.zeros((n, 1))
        phi1_x = np.zeros((n, 1))
        s_ex = np.zeros((n, 1))
        for q in range(n):
            parameters = ipf_learning_x(q, n, T, time_series, thetaEST1x, alpha0x, phi0Ex, phi1Ex, sEx, prec_learning,
                                        maxit)
            phi0_x[q] = parameters[0].copy()
            phi1_x[q] = parameters[1].copy()
            s_ex[q] = parameters[2].copy()

        phi0ESTx = np.append(phi0ESTx, phi0_x, axis=1)
        phi1ESTx = np.append(phi1ESTx, phi1_x, axis=1)
        sESTx = np.append(sESTx, s_ex, axis=1)

        phi0_diffx = phi0ESTx[:, -1] - phi0ESTx[:, -2]
        phi1_diffx = phi1ESTx[:, -1] - phi1ESTx[:, -2]
        sEST_diffx = sESTx[:, -1] - sESTx[:, -2]

        alpha1x = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                thetai = thetaEST1x[i, :].copy()
                thetaj = thetaEST1x[j, :].copy()
                Aij = np.zeros(T)
                for t in range(T):
                    Aij[t] = time_series[t][i, j]
                alpha1x[i, j] = ipf_learning_alpha_x(Aij, T, i, j, thetai, thetaj, phi0_x[i], phi1_x[i],
                                                     s_ex[i], phi0_x[j], phi1_x[j], s_ex[j],
                                                     pKw, xKw)

        precPHI = max(np.nanmax(abs(phi0_diffx)), np.nanmax(abs(phi1_diffx)), np.nanmax(abs(sEST_diffx)))
        vecPrecPhi = np.append(vecPrecPhi, precPHI)
        thetaEST0x = thetaEST1x.copy()
        phi0Ex = phi0ESTx[:, -1].copy()
        phi1Ex = phi1ESTx[:, -1].copy()
        sEx = sESTx[:, -1].copy()
        alpha0x = alpha1x.copy()

        if len(vecPrecPhi) >= 12 and (precPHI < 1) and (vecPrecPhi[iterations] > vecPrecPhi[iterations - 1]):
            precPHI = 0
        iterations = iterations + 1
    return -phi0ESTx[:, -1], phi1ESTx[:, -1], sESTx[:, -1], -thetaESTNAIVEx, -thetaEST0x, alpha0x


def dar_tgrg_simulation(n, T, theta_0=None, phi_0=None, phi_1=None, sigma=None, alpha=None):
    """
    Simulate a temporal network following the Discrete Auto-Regressive Temporally Generalized Random Graph model (:math:`DAR`-:math:`TGRG`).

    The :math:`DAR`-:math:`TGRG` model can be interpreted as a mixture of :math:`DAR` and :math:`TGRG` models where the persistence pattern associated with the copying mechanism of the :math:`DAR` model coexists with the
    node fitnesses evolving in time according to the :math:`TGRG` model.

    In the :math:`DAR`-:math:`TGRG` model [1]_ with temporal network described by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}`, each node is characterized by a latent variable :math:`\\theta_i^t`, namely the fitness of the node, which evolves in time by following a covariance stationary autoregressive process :math:`AR(1)`:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\theta_i^t = \\varphi_{0,i} + \\varphi_{1,i} \\theta_i^{t-1} + \\epsilon_i^t,`

    with :math:`\\varphi_{0,i}\\in \\mathbb{R}`, :math:`|\\varphi_{1,i}|<1`, and i.i.d. normal innovations :math:`\\epsilon_i^t\\sim \\mathcal{N}(0, \\sigma_i^2)`.

    Then, the observation equation for the network snapshot at time :math:`t` is given by :math:`\\binom{N}{2}` independent Bernoulli trials whose conditional probability is:

    :math:`\\phantom{aaaaaaaa}\\mathbb{P}(A^t| \\Theta^t, A^{t-1}, \\mathbf{\\alpha}) = \\prod_{i<j}\\left( \\alpha_{ij}\\mathbb{I}_{A^t_{ij}A^{t-1}_{ij}} + (1-\\alpha_{ij}) \\frac{e^{A^t_{ij}(\\theta_i^t + \\theta_j^t)}}{1 + e^{\\theta_i^t + \\theta_j^t}}\\right),`

    where :math:`\\Theta^t \\equiv \\{\\theta_i^t\\}_{i = 1, \\dots, n}` and :math:`\\alpha \\equiv  \\{\\alpha_{ij}\\}_{i,j = 1, \\dots, n}` with :math:`0<\\alpha_{ij}<1`.

    Parameters
    __________

    n: integer
        Number of nodes in the graph.
    T: integer
        Number of time snapshots.
    theta_0: array
        Vector with the values of :math:`\\theta_i^0`. If it is set to 'None' the null vector is used. Default: 'None'.
    phi_0: array
        Vector with the values of :math:`{\\varphi_0}_i`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    phi_1: array
        Vector with the values of :math:`{\\varphi_1}_i`. If it is set to 'None' the vector of all 0.5 is used. Default: 'None'.
    sigma: array
        Vector with the values of :math:`{\\sigma}_i`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    alpha: array_like
        Matrix that in the entry :math:`(i, j)` has the value :math:`\\alpha_{ij}`. If it is set to 'None' the matrix has 0.3 constant entries. Default 'None'.


    :return: **simulation**: (list)
        Temporal network generated according to  a :math:`DAR-TGRG` model.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Define input parameters

    .. code:: python

     >>>    n = 50
     >>>    T = 100

    Simulate the temporal network

     .. code:: python

    >>>   time_series = sm.dar_tgrg_simulation(n, T)

    References
    __________

    .. [1] Mazzarisi, P., Barucca, P., Lillo, F. and Tantari, D., 2020.
       A dynamic network model with persistent links and node-specific latent variables, with an application to the interbank market.
       European Journal of Operational Research, 281(1), pp.50-65.
       https://doi.org/10.1016/j.ejor.2019.07.024
    """
    if phi_0 is None:
        phi_0 = np.ones(n) * 0.3
    if phi_1 is None:
        phi_1 = np.ones(n) * 0.5
    if theta_0 is None:
        theta_0 = phi_0 / (1 - phi_1)
    if sigma is None:
        sigma = np.ones(n) * 0.3
    if alpha is None:
        alpha = np.ones((n, n)) * 0.3

    theta = np.zeros((n, T))
    theta[:, 0] = theta_0
    for i in range(n):
        for t in range(1, T):
            eps_i_t = np.random.normal(scale=sigma[i])
            theta[i, t] = phi_0[i] + theta[i, t - 1] * phi_1[i] + eps_i_t

    # generate adjacency matrices

    time_series = []
    A_t = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p_ij = np.exp(theta[i, 0] + theta[j, 0]) / (1 + np.exp(theta[i, 0] + theta[j, 0]))
            A_t[i, j] = np.random.choice([1, 0], 1, p=[p_ij, 1 - p_ij])
            A_t[j, i] = A_t[i, j]
    time_series.append(A_t)
    for t in range(1, T):
        A_t = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                p_ij = alpha[i, j] * time_series[t - 1][i, j] + (1 - alpha[i, j]) * np.exp(
                    theta[i, t] + theta[j, t]) / (1 + np.exp(theta[i, t] + theta[j, t]))
                A_t[i, j] = np.random.choice([1, 0], 1, p=[p_ij, 1 - p_ij])
                A_t[j, i] = A_t[i, j]
        time_series.append(A_t)
    return time_series


def dar_tgrg_directed_simulation(n, T, theta_0_in=None, theta_0_out=None, phi_0_in=None, phi_0_out=None,
                                 phi_1_in=None, phi_1_out=None, sigma_in=None, sigma_out=None, alpha=None):
    """
    Simulate a  *directed* temporal network following the Discrete Auto-Regressive Temporally Generalized Random Graph model (:math:`DAR`-:math:`TGRG`).

    The :math:`DAR`-:math:`TGRG` model [1]_ can be interpreted as a mixture of :math:`DAR` and :math:`TGRG` models where the persistence pattern associated with the copying mechanism of the :math:`DAR` model coexists with the
    node fitnesses evolving in time according to the :math:`TGRG` model.
    
    In the :math:`DAR`-:math:`TGRG` model for directed graphs, with temporal network described by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}`  with :math:`A_{ij}^t\\neq A_{ji}^t` in general, each node :math:`i` is characterized by two latent variables, :math:`\\theta_i^{t,in}` and :math:`\\theta_i^{t,out}`, both of them evolving in time by following a covariance stationary autoregressive process :math:`AR(1)`:
    
    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\theta_i^{t, in} = \\varphi_{0,i}^{in} + \\varphi_{1,i}^{in} \\theta_i^{t-1, in} + \\epsilon_i^{t, in},`

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\theta_i^{t, out} = \\varphi_{0,i}^{out} + \\varphi_{1,i}^{out} \\theta_i^{t-1, out} + \\epsilon_i^{t, out},`

     where :math:`\\varphi_{0,i}^{in}, \\varphi_{0,i}^{out}\\in \\mathbb{R}`, :math:`|\\varphi_{1,i}^{in}|, |\\varphi_{1,i}^{out}|<1`, and i.i.d. normal innovations :math:`\\epsilon_i^{t, in}\\sim \\mathcal{N}(0, {\\sigma_i^{in}}^{2})` and :math:`\\epsilon_i^{t, out}\\sim \\mathcal{N}(0, {\\sigma_i^{out}}^2)`.
    
    Then, the observation equation for the network snapshot at time :math:`t` is given by :math:`N(N-1)` independent Bernoulli trials whose conditional probability is:
    
    :math:`\\phantom{aaaa}\\mathbb{P}(A^t| \\Theta^t, A^{t-1}, \\mathbf{\\alpha}) = \\prod_{i<j}\\left( \\alpha_{ij}\\mathbb{I}_{A^t_{ij}A^{t-1}_{ij}} + (1-\\alpha_{ij}) \\frac{e^{A^t_{ij}(\\theta_i^{t, out} + \\theta_j^{t, in})}}{1 + e^{\\theta_i^{t, out} + \\theta_j^{t, in}}}\\right)`,

    
    where :math:`\\Theta^t \\equiv \\{\\theta_i^{t, in}, \\theta_i^{t, out}\\}_{i = 1, \\dots, n}` and :math:`\\alpha \\equiv  \\{\\alpha_{ij}\\}_{i,j = 1, \\dots, n}` with :math:`0<\\alpha_{ij}<1`.

    Parameters
    __________

    n: integer
        Number of nodes in the graph.
    T: integer
        Number of time snapshots.
    theta_0_in: array
        Vector with the values of :math:`{\\theta_i^0}^{in}`. If it is set to 'None' the null vector is used. Default: 'None'.
    theta_0_out: array
        Vector with the values of :math:`{\\theta_i^0}^{out}`. If it is set to 'None' the null vector is used. Default: 'None'.
    phi_0_in: array
        Vector with the values of :math:`{{\\varphi_0}_i}^{in}`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    phi_0_out: array
        Vector with the values of :math:`{{\\varphi_0}_i}^{out}`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    phi_1_in: array
        Vector with the values of :math:`{{\\varphi_1}_i}^{in}`. If it is set to 'None' the vector of all 0.5 is used. Default: 'None'.
    phi_1_out: array
        Vector with the values of :math:`{{\\varphi_1}_i}^{out}`. If it is set to 'None' the vector of all 0.5 is used. Default: 'None'.
    sigma_in: array
        Vector with the values of :math:`{{\\sigma}_i}^{in}`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    sigma_out: array
        Vector with the values of :math:`{{\\sigma}_i}^{out}`. If it is set to 'None' the vector of all 0.3 is used. Default: 'None'.
    alpha: array_like
        Matrix that in the entry :math:`(i, j)` has the value :math:`\\alpha_{ij}`. If it is set to 'None' the matrix has 0.3 constant entries. Default: 'None'.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Define input parameters

    .. code:: python

     >>>    n = 60
     >>>    T = 150
     >>>    phi0 = np.ones(n) * 0.2


    Create temporal network

    .. code:: python

     >>>    time_series = dar_tgrg_directed_simulation(n, T, phi_0_in=phi0, phi_0_out=phi0)

    :return: **simulation**: (list)
        Directed temporal network generated according to  a :math:`DAR-TGRG` model.

    References
    __________

    .. [1] Mazzarisi, P., Barucca, P., Lillo, F. and Tantari, D., 2020.
       A dynamic network model with persistent links and node-specific latent variables, with an application to the interbank market.
       European Journal of Operational Research, 281(1), pp.50-65.
       https://doi.org/10.1016/j.ejor.2019.07.024

    """

    # generate latent variables
    if phi_0_in is None:
        phi_0_in = np.ones(n) * 0.3
    if phi_0_out is None:
        phi_0_out = np.ones(n) * 0.3
    if phi_1_in is None:
        phi_1_in = np.ones(n) * 0.5
    if phi_1_out is None:
        phi_1_out = np.ones(n) * 0.5
    if theta_0_in is None:
        theta_0_in = phi_0_in / (1 - phi_1_in)
    if theta_0_out is None:
        theta_0_out = phi_0_out / (1 - phi_1_out)
    if sigma_in is None:
        sigma_in = np.ones(n) * 0.3
    if sigma_out is None:
        sigma_out = np.ones(n) * 0.3
    if alpha is None:
        alpha = np.ones((n, n)) * 0.3

    phi_0 = np.zeros(2 * n)
    phi_0[:n] = phi_0_out.copy()
    phi_0[n:] = phi_0_in.copy()

    phi_1 = np.zeros(2 * n)
    phi_1[:n] = phi_1_out.copy()
    phi_1[n:] = phi_1_in.copy()

    sigma = np.zeros(2 * n)
    sigma[:n] = sigma_out.copy()
    sigma[n:] = sigma_in.copy()

    theta = np.zeros((2 * n, T))
    theta[:n, 0] = theta_0_out.copy()
    theta[n:, 0] = theta_0_in.copy()

    for i in range(2 * n):
        for t in range(1, T):
            eps_i_t = np.random.normal(scale=sigma[i])
            theta[i, t] = phi_0[i] + theta[i, t - 1] * phi_1[i] + eps_i_t

    # generate adjacency matrices

    time_series = []
    A_t = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j != i:
                p_ij = np.exp(theta[i, 0] + theta[j + n, 0]) / (1 + np.exp(theta[i, 0] + theta[j + n, 0]))
                A_t[i, j] = np.random.choice([1, 0], 1, p=[p_ij, 1 - p_ij])
    time_series.append(A_t)
    for t in range(1, T):
        A_t = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j != i:
                    p_ij = alpha[i, j] * time_series[t - 1][i, j] + (1 - alpha[i, j]) * \
                           np.exp(theta[i, t] + theta[j + n, t]) / (1 + np.exp(theta[i, t] + theta[j + n, t]))
                    A_t[i, j] = np.random.choice([1, 0], 1, p=[p_ij, 1 - p_ij])
        time_series.append(A_t)
    return time_series


def grad_x_i_directed(theta, i, At, At1, thetaV, alpha, phi0, phi1, sigma, thetaP):
    F = np.exp(-theta - thetaV) / (1 + np.exp(- theta - thetaV))
    IAi = np.array(1 * (At[i, :] == At1[i, :]))
    IAi[i] = 0
    Ait = At[i, :]
    ai = alpha[i, :]
    PAi = np.exp(-Ait * (theta + thetaV)) / (1 + np.exp(-theta - thetaV))
    g = (((1 - ai) * PAi) / (ai * IAi + (1 - ai) * PAi)) * (-Ait + F)
    g[i] = 0
    gi = np.sum(g) - (theta - phi0 - phi1 * thetaP) / (sigma ** 2)
    return gi


def kernel_theta_0_x_directed(x, y, i, Ati, IAti, ai, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    f = np.zeros((1, L))
    for k in range(L):
        F = ai * IAti + (1 - ai) * (np.exp(-Ati * (x[k] + y)) / (1 + np.exp(-x[k] - y)))
        F[i] = 1
        f[k] = np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return f


def kernel_theta_1_x_directed(x, y, i, Ati, IAti, ai, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    g1 = np.zeros((1, L))
    for k in range(L):
        F = ai * IAti + (1 - ai) * (np.exp(-Ati * (x[k] + y)) / (1 + np.exp(-x[k] - y)))
        F[i] = 1
        g1[k] = x[k] * np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    return g1


def kernel_theta_2_x_directed(x, y, i, Ati, IAti, ai, phi0, phi1, s, xp):
    x = np.array([x])
    L = len(x)
    g2 = np.zeros((1, L))
    for k in range(L):
        F = ai * IAti + (1 - ai) * (np.exp(-Ati * (x[k] + y)) / (1 + np.exp(-x[k] - y)))
        F[i] = 1
        g2[k] = (x[k] ** 2) * np.prod(F) * np.exp(-(x[k] - phi0 - phi1 * xp) ** 2 / (2 * s ** 2)) / (
                np.sqrt(2 * np.pi) * s)
    return g2


def ipf_learning_alpha_x_directed(At, T, i, j, thetai, thetaj, phi0xi, phi1xi, sexi, phi0xj, phi1xj, sexj, pKw, xKw):
    bound0 = 1e-3
    bound1 = 0.999
    Apost = At[1:].copy()
    Apre = At[:-1].copy()
    IA = 1 * (Apost == Apre)
    KernelA = np.zeros((T - 1, 1))
    for t in range(1, T):
        KernelA[t - 1] = np.trapz((1 / 2) * pKw * (1 / np.sqrt(1 + xKw * (sexi ** 2 + sexj ** 2)))
                                  * np.exp(((1 - 2 * At[t]) ** 2 * (sexi ** 2 + sexj ** 2) +
                                            4 * (phi0xi + phi1xi * thetai[t - 1] + phi0xj + phi1xj * thetaj[t - 1])
                                            * (1 - 2 * At[t] - xKw *
                                               (phi0xi + phi1xi * thetai[t - 1] + phi0xj + phi1xj * thetaj[t - 1]))) /
                                           (8 * (1 + xKw * (sexi ** 2 + sexj ** 2)))), xKw)

    ga0 = np.sum(IA / (bound0 * IA + (1 - bound0) * KernelA) - KernelA / (bound0 * IA + (1 - bound0) * KernelA))
    ga1 = np.sum(IA / (bound1 * IA + (1 - bound1) * KernelA) - KernelA / (bound1 * IA + (1 - bound1) * KernelA))
    if ga0 * ga1 < 0:
        alphaij = root_scalar(auxiliary_alpha_x, args=(IA, KernelA), bracket=(bound0, bound1)).root
    else:
        alphaij = 0
    return alphaij


def ipf_learning_x_directed(q, n, T, A, thetaE, alphaE, phi0E, phi1E, sE, prec_learning, ite_learning):
    BOUND = 30
    if q < n:
        xhat = thetaE[q, :].copy()
        yhat = thetaE[n:, :].copy()
    else:
        xhat = thetaE[q, :].copy()
        yhat = thetaE[:n, :].copy()
    phi0 = np.zeros((3, 1))
    phi0[0] = phi0E[q].copy()
    phi0[1] = phi1E[q].copy()
    phi0[2] = sE[q].copy()
    phi1 = np.zeros((3, 1))
    temp_prec = 1
    iterations = 1

    while (temp_prec > prec_learning) and (iterations <= ite_learning):
        zt = np.zeros((T - 1, 1))
        g1t = np.zeros((T - 1, 1))
        g2t = np.zeros((T - 1, 1))
        for t in range(1, T):
            y = yhat[:, t].copy()
            if q < n:
                i = q
                Ati = A[t][i, :].copy()
                IAti = 1 * (A[t][i, :] == A[t - 1][i, :])
                ai = alphaE[i, :].copy()
            else:
                i = q - n
                Ati = A[t][:, i].copy()
                IAti = 1 * (A[t][:, i] == A[t - 1][:, i])
                ai = alphaE[:, i].copy()
            zt[t - 1] = quad(kernel_theta_0_x_directed, -BOUND, BOUND, args=(y, i, Ati, IAti, ai, phi0[0], phi0[1],
                                                                             phi0[2], xhat[t - 1]), points=0)[0]
            g1t[t - 1] = quad(kernel_theta_1_x_directed, -BOUND, BOUND, args=(y, i, Ati, IAti, ai, phi0[0], phi0[1],
                                                                              phi0[2], xhat[t - 1]), points=0)[0]
            g2t[t - 1] = quad(kernel_theta_2_x_directed, -BOUND, BOUND, args=(y, i, Ati, IAti, ai, phi0[0], phi0[1],
                                                                              phi0[2], xhat[t - 1]), points=0)[0]
        tphi = np.mean(g1t / zt)
        t2phi = np.mean(g2t / zt)
        that = np.mean(xhat[:T - 1])
        t2hat = np.mean(xhat[:T - 1] ** 2)
        thattphi = np.mean(np.reshape(xhat[:T - 1], (1, T - 1)) * np.reshape(g1t / zt, (1, T - 1)))
        phi1[1] = (thattphi - that * tphi) / (t2hat - that ** 2)
        phi1[0] = tphi - that * phi1[1]
        phi1[2] = np.sqrt(abs(t2phi + phi1[0] ** 2 + t2hat * phi1[1] ** 2 - 2 * tphi * phi1[0] - 2 * thattphi * phi1[1]
                              + 2 * that * phi1[0] * phi1[1]))
        temp_prec = np.max(abs((phi1 - phi0) / phi0))
        phi0 = phi1.copy()
        iterations = iterations + 1
    parameters = phi1.copy()
    return parameters


def dar_tgrg_directed(time_series, tol=1e-2, maxit=1e2):
    """
    Estimate, by an expectation-maximization algorithm, the parameters of the Discrete Auto Regressive Temporally Generalized Random Graph model (:math:`DAR`-:math:`TGRG`).

    The :math:`DAR`-:math:`TGRG` model [1]_ can be interpreted as a mixture of :math:`DAR` and :math:`TGRG` models where the persistence pattern associated with the copying mechanism of the :math:`DAR` model coexists with the
    node fitnesses evolving in time according to the :math:`TGRG` model.

    In the :math:`DAR`-:math:`TGRG` model for directed graphs, with temporal network described by a time series of adjacency matrices :math:`\\{A_{ij}^t\\}_{i,j=1,\\ldots, n}^{t=1,\\ldots,s}`, each node :math:`i` is characterized by two latent variables, :math:`\\theta_i^{t,in}` and :math:`\\theta_i^{t,out}`, both of them evolving in time by following a covariance stationary autoregressive process :math:`AR(1)`:

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\theta_i^{t, in} = \\varphi_{0,i}^{in} + \\varphi_{1,i}^{in} \\theta_i^{t-1, in} + \\epsilon_i^{t, in},`

    :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\theta_i^{t, out} = \\varphi_{0,i}^{out} + \\varphi_{1,i}^{out} \\theta_i^{t-1, out} + \\epsilon_i^{t, out},`

     where :math:`\\varphi_{0,i}^{in}, \\varphi_{0,i}^{out}\\in \\mathbb{R}`, :math:`|\\varphi_{1,i}^{in}|, |\\varphi_{1,i}^{out}|<1`, and i.i.d. normal innovations :math:`\\epsilon_i^{t, in}\\sim \\mathcal{N}(0, {\\sigma_i^{in}}^{2})` and :math:`\\epsilon_i^{t, out}\\sim \\mathcal{N}(0, {\\sigma_i^{out}}^2)`.

    Then, the observation equation for the network snapshot at time :math:`t` is given by :math:`N(N-1)` independent Bernoulli trials whose conditional probability is:

    :math:`\\phantom{aaaa}\\mathbb{P}(A^t| \\Theta^t, A^{t-1}, \\mathbf{\\alpha}) = \\prod_{i<j}\\left( \\alpha_{ij}\\mathbb{I}_{A^t_{ij}A^{t-1}_{ij}} + (1-\\alpha_{ij}) \\frac{e^{A^t_{ij}(\\theta_i^{t, out} + \\theta_j^{t, in})}}{1 + e^{\\theta_i^{t, out} + \\theta_j^{t, in}}}\\right)`,


    where :math:`\\Theta^t \\equiv \\{\\theta_i^{t, in}, \\theta_i^{t, out}\\}_{i = 1, \\dots, n}` and :math:`\\alpha \\equiv  \\{\\alpha_{ij}\\}_{i,j = 1, \\dots, n}` with :math:`0<\\alpha_{ij}<1`.

    Parameters
    __________
    time_series: List object
        List of adjacency matrices [:math:`A_1, \\dots, A_T`] with binary entries and zero diagonal.
    tol: float
         Relative error of the estimated parameters. Default: 1e-2.
    maxit: integer
        Maximum number of iterations in the learning process. Default: 1e2.

    Returns
    _______

    phi_0: array
        Vector of length :math:`2n` with the estimated values of :math:`\\varphi_0`. The first :math:`n` entries contain :math:`\\varphi_0^{out}`, while the last :math:`n` entries contain :math:`\\varphi_0^{in}`.
    phi_1: array
        Vector of length :math:`2n` with the estimated values of :math:`\\varphi_1`. The first :math:`n` entries contain :math:`\\varphi_1^{out}`, while the last :math:`n` entries contain :math:`\\varphi_1^{in}`.
    sigma: array
        Vector of length :math:`2n` with the estimated values of :math:`\\sigma`. The first :math:`n` entries contain :math:`\\sigma^{out}:`, while the last :math:`n` entries contain :math:`\\sigma^{in}`.
    theta_naive: array_like
        Matrix of size :math:`2n \\times T`, that in the entry :math:`(i, t)` has a naive estimation of the :math:`\\theta` parameters. The first :math:`2n` rows contain the values of the :math:`{\\theta^{t}_i}^{out}` while the last :math:`n` rows contain the :math:`{\\theta^{t}_i}^{in}`.
    theta: array_like
        Matrix of size :math:`2n \\times T`, that in the entry :math:`(i, t)` has the estimation of the :math:`\\theta` parameters. The first :math:`2n` rows contain the values of the :math:`{\\theta^{t}_i}^{out}` while the last :math:`n` rows contain the :math:`{\\theta^{t}_i}^{in}`.
    alpha: array_like
        Matrix with the estimated values of :math:`\\alpha_{ij}`.

    Examples
    ________

    .. code:: python

     >>>  from networksns import statistical_models as sm
     >>>  import numpy as np

    Define input parameters

    .. code:: python

     >>>    n = 60
     >>>    T = 150
     >>>    phi0 = np.ones(n) * 0.2


    Create temporal network

    .. code:: python

     >>>    time_series = dar_tgrg_directed_simulation(n, T, phi_0_in=phi0, phi_0_out=phi0)

    Estimate the :math:`DAR`-:math:`TGRG` model parameters

      .. code:: python

     >>>    phi_0, phi_1, sigma, theta_naive, theta, alpha = sm.tgrg(time_series)



    References
    __________

    .. [1] Mazzarisi, P., Barucca, P., Lillo, F. and Tantari, D., 2020.
       A dynamic network model with persistent links and node-specific latent variables, with an application to the interbank market.
       European Journal of Operational Research, 281(1), pp.50-65.
       https://doi.org/10.1016/j.ejor.2019.07.024

    """
    with open('polya_gamma_points', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    xKw = np.zeros(len(data))
    for i in range(len(data)):
        xKw[i] = float(data[i][1])

    with open('polya_gamma_values', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    pKw = np.zeros(len(data))
    for i in range(len(data)):
        pKw[i] = float(data[i][1])

    n = np.shape(time_series[0])[0]
    T = len(time_series)
    precision_0 = tol
    precision_1 = tol
    precision_phi = tol
    prec_learning = tol

    thetaESTNAIVEx = np.zeros((2 * n, T))

    for t in range(T):
        A_t = time_series[t]
        k_out = np.sum(A_t, 1)
        k_in = np.sum(A_t, 0)
        k = np.append(np.reshape(k_out, (n, 1)), np.reshape(k_in, (n, 1)))
        ks = k.copy()
        ks[k == 0] = 1e-4
        x_0 = np.random.rand(2 * n, 1)
        x = np.zeros((2 * n, 1))
        temp_pre = 1
        temp_ite = 1

        while temp_pre > precision_0 and temp_ite <= 10 * maxit:
            xfOut = x_0[:n].copy()
            xfIn = x_0[n:].copy()
            matrix_gi = (np.ones((n, 1)) * np.reshape(xfIn, (1, n))) / \
                        (1 + (np.reshape(xfOut, (n, 1)) * np.reshape(xfIn, (1, n))))
            matrix_gj = (np.ones((n, 1)) * np.reshape(xfOut, (1, n))) / \
                        (1 + (np.reshape(xfIn, (n, 1)) * np.reshape(xfOut, (1, n))))
            matrix_gi = matrix_gi - np.diag(np.diag(matrix_gi))
            matrix_gj = matrix_gj - np.diag(np.diag(matrix_gj))
            matrix_g = np.zeros((2 * n, n))
            matrix_g[:n, :] = matrix_gi.copy()
            matrix_g[n:, :] = matrix_gj.copy()
            x = ks / np.sum(matrix_g, 1)
            g = (x - x_0) / x_0
            g[k == 0] = 0
            temp_pre = np.max(abs(g)).copy()
            temp_ite = temp_ite + 1
            x_0 = x.copy()

        x_out = x[:n].copy()
        x_in = x[n:].copy()
        x_out = x_out / x_out[0]
        x_in = x_in * x_out[0]
        x = np.append(np.reshape(x_out, (n, 1)), np.reshape(x_in, (n, 1)))
        thetaESTNAIVEx[:, t] = -np.log(x)
        if temp_ite > 10 * maxit:
            warnings.warn('Naive estimation at time %d may be inaccurate: reached maximum number of iterations, ' % t)
    phi0_x = np.zeros((2 * n, 1))
    phi1_x = np.zeros((2 * n, 1))
    s_ex = np.zeros((2 * n, 1))

    for q in range(1, 2 * n):
        y = thetaESTNAIVEx[q, :]
        model = ARIMA(y, order=[1, 0, 0], trend='c')
        fit = model.fit()
        model_params = fit.params
        phi0_x[q] = model_params[0]
        phi1_x[q] = model_params[1]
        s_ex[q] = np.sqrt(model_params[2])

    phi0ESTx = phi0_x.copy()
    phi1ESTx = phi1_x.copy()
    sESTx = s_ex.copy()

    alpha0NAIVEx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                vecA = np.zeros(T)
                for t in range(T):
                    vecA[t] = time_series[t][i, j]
                chiest, rhoest = dar1_x_starting_point(vecA)
                alpha0NAIVEx[i, j] = rhoest

    thetaEST0x = thetaESTNAIVEx.copy()
    phi0Ex = phi0ESTx[:, -1].copy()
    phi1Ex = phi1ESTx[:, -1].copy()
    sEx = sESTx[:, -1].copy()
    alpha0x = alpha0NAIVEx.copy()
    precPHI = 1
    vecPrecPhi = []
    iterations = 0

    while (precPHI > precision_phi) and (iterations <= maxit):

        thetaEST1x = np.zeros((2 * n, T))
        thetaEST1x[:, 0] = thetaEST0x[:, 0].copy()
        for t in range(1, T):
            At = time_series[t].copy()
            At1 = time_series[t - 1].copy()
            theta0x = thetaEST0x[:, t].copy()
            temp_pre = 1
            temp_ite = 0
            theta1x = np.zeros((2 * n, 1))
            while (temp_pre > precision_1) and (temp_ite <= 10 * maxit):
                theta0xOUT = theta0x[:n].copy()
                theta0xIN = theta0x[n:].copy()
                theta1x = np.zeros((2 * n, 1))
                for i in range(1, 2 * n):
                    bound_0 = -30
                    bound_1 = 30
                    if i < n:
                        theta_bound_0 = grad_x_i_directed(bound_0, i, At, At1, theta0xIN, alpha0x, phi0Ex[i],
                                                          phi1Ex[i], sEx[i], thetaEST0x[i, t - 1])
                        theta_bound_1 = grad_x_i_directed(bound_1, i, At, At1, theta0xIN, alpha0x, phi0Ex[i],
                                                          phi1Ex[i], sEx[i], thetaEST0x[i, t - 1])
                        if theta_bound_0 * theta_bound_1 < 0:
                            theta1x[i] = root_scalar(grad_x_i_directed,
                                                     args=(i, At, At1, theta0xIN, alpha0x, phi0Ex[i],
                                                           phi1Ex[i], sEx[i], thetaEST0x[i, t - 1]),
                                                     bracket=[bound_0, bound_1]).root
                        else:
                            theta1x[i] = thetaEST0x[i, t - 1]
                    else:
                        theta_bound_0 = grad_x_i_directed(bound_0, i - n, At.transpose(), At1.transpose(),
                                                          theta0xOUT, alpha0x.transpose(),
                                                          phi0Ex[i],
                                                          phi1Ex[i], sEx[i], thetaEST0x[i, t - 1])
                        theta_bound_1 = grad_x_i_directed(bound_1, i - n, At.transpose(), At1.transpose(),
                                                          theta0xOUT, alpha0x.transpose(),
                                                          phi0Ex[i],
                                                          phi1Ex[i], sEx[i], thetaEST0x[i, t - 1])
                        if theta_bound_0 * theta_bound_1 < 0:
                            theta1x[i] = root_scalar(grad_x_i_directed,
                                                     args=(i - n, At.transpose(), At1.transpose(),
                                                           theta0xOUT, alpha0x.transpose(),
                                                           phi0Ex[i],
                                                           phi1Ex[i], sEx[i], thetaEST0x[i, t - 1]),
                                                     bracket=[bound_0, bound_1]).root
                        else:
                            theta1x[i] = thetaEST0x[i, t - 1]
                temp_pre = np.max(abs(theta1x - theta0x)).copy()
                theta0x = theta1x.copy()
                temp_ite = temp_ite + 1

            if temp_ite > 10 * maxit:
                warnings.warn('The estimation of theta at time %d may be inaccurate: reached maximum iterations , ' % t)

            thetaEST1x[:, t] = np.reshape(theta1x, 2 * n).copy()
        phi0_x = np.zeros((2 * n, 1))
        phi1_x = np.zeros((2 * n, 1))
        s_ex = np.zeros((2 * n, 1))

        for q in range(1, 2 * n):
            parameters = ipf_learning_x_directed(q, n, T, time_series, thetaEST1x, alpha0x, phi0Ex, phi1Ex, sEx,
                                                 prec_learning, maxit)
            phi0_x[q] = parameters[0].copy()
            phi1_x[q] = parameters[1].copy()
            s_ex[q] = parameters[2].copy()

        phi0ESTx = np.append(phi0ESTx, phi0_x, axis=1)
        phi1ESTx = np.append(phi1ESTx, phi1_x, axis=1)
        sESTx = np.append(sESTx, s_ex, axis=1)

        phi0_diffx = (phi0ESTx[1:, -1] - phi0ESTx[1:, -2]) / phi0ESTx[1:, -2]
        phi1_diffx = (phi1ESTx[1:, -1] - phi1ESTx[1:, -2]) / phi1ESTx[1:, -2]
        sEST_diffx = (sESTx[1:, -1] - sESTx[1:, -2]) / sESTx[1:, -2]

        alpha1x = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    thetai = thetaEST1x[i, :].copy()
                    thetaj = thetaEST1x[n + j, :].copy()
                    Aij = np.zeros(T)
                    for t in range(T):
                        Aij[t] = time_series[t][i, j]
                    alpha1x[i, j] = ipf_learning_alpha_x_directed(Aij, T, i, j, thetai, thetaj, phi0_x[i], phi1_x[i],
                                                                  s_ex[i], phi0_x[n + j], phi1_x[n + j], s_ex[n + j],
                                                                  pKw, xKw)

        precPHI = max(np.nanmax(abs(phi0_diffx)), np.nanmax(abs(phi1_diffx)), np.nanmax(abs(sEST_diffx)))
        approxPHI = max(np.mean(abs(phi0_diffx)), np.mean(abs(phi1_diffx)), np.mean(abs(sEST_diffx)))
        vecPrecPhi = np.append(vecPrecPhi, precPHI)
        thetaEST0x = thetaEST1x.copy()
        phi0Ex = phi0ESTx[:, -1].copy()
        phi1Ex = phi1ESTx[:, -1].copy()
        sEx = sESTx[:, -1].copy()
        alpha0x = alpha1x.copy()

        if len(vecPrecPhi) >= 16 and (approxPHI < 1) and (vecPrecPhi[iterations] > vecPrecPhi[iterations - 1]):
            precPHI = 0
        if len(vecPrecPhi) >= 14 and (approxPHI < 1e-1) and (vecPrecPhi[iterations] > vecPrecPhi[iterations - 1]):
            precPHI = 0
        if len(vecPrecPhi) >= 12 and (approxPHI < tol) and (vecPrecPhi[iterations] > vecPrecPhi[iterations - 1]):
            precPHI = 0
        iterations = iterations + 1
    return -phi0ESTx[:, -1], phi1ESTx[:, -1], sESTx[:, -1], -thetaESTNAIVEx, -thetaEST0x, alpha0x
