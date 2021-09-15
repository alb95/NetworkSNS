import networkx as nx
import dynetx as dn
import numpy as np
from scipy.sparse.linalg import expm_multiply, cg
from scipy.sparse.linalg import norm as scipy_norm
from scipy.sparse import identity, diags, csr_matrix, bmat, kron
from numpy.linalg import norm
from numpy import ones, zeros, arange, inf
from math import exp
import bisect
import warnings

# ======================================================================================
#  AUXILIARY FUNCTIONS: lanczos_symmetric_exponential, lanczos_exponential, graph_slice
# ======================================================================================


def exponential_symmetric_quadrature(A, u, tol=1e-7, maxit=50):

    """

    Computes :math:`q=u^Te^Au`.
    The computation is done by means of Lanczos method according to [1]_.

    Parameters
    __________
    A: array_like
        sparse/dense symmetric matrix.
    u: array
        vector.
    tol: float,optional
        tolerance for convergence, relative accuracy, default: 1e-7.
    maxit: integer, optional
     maximum number of Lanczos iterations, default: 50.


    :return: **q**: (float)
     value of the quadratic form :math:`u^Te^Au`.


    Examples
    ________
    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import numpy as np

    Create symmetric matrix :math:`A`

    .. code:: python

     >>>    A = np.arange(0, 9, 1)
     >>>    A = A.reshape(3, 3)
     >>>    A = A + A.transpose()
            array([[ 0,  4,  8],
                   [ 4,  8, 12],
                   [ 8, 12, 16]])

    Create vector :math:`u`

        .. code:: python

    >>>     u = np.arange(0, 3)
            array([0, 1, 2])

    Compute :math:`q=u^T e^A u`.

     >>>    q = cm.exponential_symmetric_quadrature(A, u)

    References
    ----------
    .. [1] G. H. Golub, and G. Meurant (2010)
           "Matrices, Moments and Quadrature with Applications",
           Princeton University Press, Princeton, NJ.

    """
    quadrature = 1
    quadrature_old = 2
    old_err = 1
    err = 1
    omega = []
    gamma = []
    u_norm = norm(u)
    if u_norm == 0:
        return 0
    else:
        x_0 = u/u_norm
        #  computing Lanczos matrix J
        omega.append(x_0.dot(A.dot(x_0)))
        r = A.dot(x_0) - omega[0] * x_0
        r_norm = norm(r)
        if r_norm == 0:
            return exp(omega[0]) * u_norm ** 2
        gamma.append(r_norm)
        x_1 = r / r_norm
        it = 1  # iterations
        while err > tol and old_err > tol and it < maxit:
            z = A.dot(x_1) - gamma[it - 1] * x_0
            omega.append(x_1.dot(z))
            x_2 = z - omega[it] * x_1
            if norm(x_2) == 0:
                gamma.append(0)  # variable used only to avoid deletion in line: eta = gamma[:-1]
                eta = gamma[:-1]
                J = diags(omega)
                J = J + diags(eta, 1)
                J = J + diags(eta, -1)
                e_1 = zeros(len(omega))
                e_1[0] = 1
                quadrature = e_1.dot(expm_multiply(J, e_1)) * (norm(u)) ** 2
                break
            gamma.append(norm(x_2))
            x_0 = x_1
            x_1 = x_2 / gamma[it]
            eta = gamma[:-1]
            J = diags(omega)
            J = J + diags(eta, 1)
            J = J + diags(eta, -1)
            e_1 = zeros(len(omega))
            e_1[0] = 1
            quadrature_very_old = quadrature_old
            quadrature_old = quadrature
            quadrature = e_1.dot(expm_multiply(J, e_1))*(norm(u))**2
            old_err = err
            err = max(abs((quadrature_old-quadrature))/quadrature_old,
                      abs((quadrature_very_old-quadrature_old))/quadrature_very_old)
            it = it+1
    return quadrature


def exponential_quadrature(A, u, v, tol=1e-7, maxit=50):

    """
    Computes :math:`q=u^T e^A v`. For the computation polarization rule and Lanczos iteration are used [1]_.

    Parameters
    __________
    A: array_like
        sparse/dense symmetric matrix.
    u: array
        vector.
    v: array
        vector.
    tol: float,optional
        tolerance for convergence, relative accuracy, default: 1e-7.
    maxit: integer, optional
        maximum number of Lanczos iterations, default: 50.


    :return: **q**: (float)
        value of the bilinear form :math:`u^Te^Av`.

    Examples
    ________
    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import numpy as np

    Create symmetric matrix :math:`A`

    .. code:: python

     >>>    A = np.arange(0, 9, 1)
     >>>    A = A.reshape(3, 3)
     >>>    A = A + A.transpose()
            array([[ 0,  4,  8],
                   [ 4,  8, 12],
                   [ 8, 12, 16]])

    Create vectors :math:`u` and :math:`v`

    .. code:: python

     >>>    u = np.arange(0, 3)
            array([0, 1, 2])
     >>>    v = np.array([2,5,2])
            array([2, 5, 2])

    Compute :math:`q=u^T e^A v`

     >>>    q = cm.exponential_quadrature(A, u, v)

    References
    ----------
    .. [1] G. H. Golub, and G. Meurant (2010)
           "Matrices, Moments and Quadrature with Applications",
           Princeton University Press, Princeton, NJ.
    """

    quadrature_sum = exponential_symmetric_quadrature(A, u+v, tol, maxit)
    quadrature_difference = exponential_symmetric_quadrature(A, u-v, tol, maxit)

    # polarization formula
    quadrature = (quadrature_sum - quadrature_difference)/4
    return quadrature


def graph_slice(G, t):
    """
    extract a slice/snapshot of the Dynamic graph, that is a snapshot of the graph :math:`G` at time :math:`t` in \
    NetworkX format.

    Parameters
    __________

    G: DynGraph or DynDiGraph object.
    t: float; snapshot time.

    :return s: (NetworkX Graph object)
        Snapshot at time :math:`t` of :math:`G`.

    """

    """
    Examples
    --------
    >>> import dynetx as dn
    >>> import networkx as nx
    >>> G = dn.DynGraph()
    >>> G.add_interaction(1, 2, 2)
    >>> G.add_interaction(1, 2, 2, e=6)
    >>> G.add_interaction(1, 2, 7, e=11)
    >>> h = graph_slice(G, 3)
    >>> print(nx.adjacency_matrix(h))
    >>>    (0, 1)	1
    >>>    (1, 0)	1
    """

    node_list = G.nodes()
    slice_t = list(dn.interactions(G, t=t))
    edge_list = ([e[0], e[1]] for e in slice_t)
    sliced_graph = nx.Graph()
    sliced_graph.add_nodes_from(node_list)
    sliced_graph.add_edges_from(edge_list)
    return sliced_graph


# =============================
# UNDIRECTED NETWORK FUNCTIONS
# =============================


def subgraph_centrality(G, t=1, tol=1e-7, maxit=50):

    """
    Computes the subgraph centrality of all the nodes in graph :math:`G`.

    The subgraph centrality of the :math:`i^{th}` node is given by :math:`[e^{tA}]_{ii}=e_i^T (e^{tA})e_i`,
    where :math:`e_i` and :math:`A` denote respectively the :math:`i^{th}` vector of the canonical basis and the adjacency matrix of the graph [1]_.


    Parameters
    __________
    G: Graph or DiGraph object
        a graph.
    t: scalar, optional
     when exponentiating multiply the adjacency matrix by t, default: 1.
    tol: float,optional
     tolerance for convergence, relative accuracy, default: 1e-7.
    maxit: integer, optional
     maximum number of Lanczos iterations, default: 50.


    :return: **sc** (dict)  subgraph centrality of all nodes in :math:`G`.

    Examples
    ________
    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`.

    .. code:: python

     >>>    G = nx.Graph()
     >>>    G.add_edge(1, 2)
     >>>    G.add_edge(2, 3)
            EdgeView([(1, 2), (2, 3)])

    Compute the subgraph centrality.

     >>>    sc = cm.subgraph_centrality(G)


    References
    ----------
    .. [1] Ernesto Estrada and Juan A. Rodríguez-Velázquez (2005)
           Subgraph centrality in complex networks,
           Physical Review, Volume 71, Issue 5,
           https://doi.org/10.1103/PhysRevE.71.056103
    """

    n = G.number_of_nodes()
    node_list = list(G.nodes)
    subgraph_centrality = np.zeros(n)
    for i in range(n):
        subgraph_centrality[i] = node_subgraph_centrality(G, node_list[i], t, tol, maxit)
    centrality = dict(zip(node_list, subgraph_centrality))
    return centrality


def node_subgraph_centrality(G, u, t=1, tol=1e-7, maxit=50):

    """
    Computes the subgraph centrality of node :math:`u`.

    If node :math:`u` is the :math:`i^{th}` node of the graph, the subgraph centrality of node :math:`u` is given by :math:`[e^{tA}]_{ii}=e_i^T (e^{tA})e_i`,
    where :math:`e_i` and :math:`A` denote respectively the :math:`i^{th}` vector of the canonical basis and the adjacency matrix of the graph [1]_.

    Parameters
    __________

    G: Graph object
        a graph.
    u: node_id
        node in G.
    t: scalar, optional
     when exponentiating multiply the adjacency matrix by t, default: 1.
    tol: float,optional
     tolerance for convergence, relative accuracy, default: 1e-7.
    maxit: integer, optional
     maximum number of Lanczos iterations, default: 50.

     :return: **sc_u** (float) subgraph centrality of node :math:`u`.

    Examples
    ________
    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`.

    .. code:: python

     >>>    G = nx.Graph()
     >>>    G.add_edge(1, 'u')
     >>>    G.add_edge('u', 2)
            EdgeView([(1, 'u'), ('u', 2)])

    Compute the node total communicability

     >>>    sc_u = cm.node_subgraph_centrality(G, 'u')

    References
    ----------
    .. [1] Ernesto Estrada and Juan A. Rodríguez-Velázquez (2005)
           Subgraph centrality in complex networks,
           Physical Review, Volume 71, Issue 5,
           https://doi.org/10.1103/PhysRevE.71.056103
    """

    n = G.number_of_nodes()
    node_list = G.nodes
    enumerated_nodes = dict(zip(node_list, arange(n)))
    node_position = enumerated_nodes[u]
    e_node = zeros(n)
    e_node[node_position] = 1
    Adj = nx.adjacency_matrix(G)
    if t != 1:
        Adj = Adj * t
    subgraph_centrality = exponential_symmetric_quadrature(Adj, e_node, tol, maxit)
    return subgraph_centrality


def total_communicability(G, t=1):

    """
    Computes the total communicability of all the nodes of a graph :math:`G`.

    Total communicability is defined as the row sums of the exponential of the adjacency matrix [1]_, so denoting with
    :math:`A` the adjacency matrix of graph :math:`G` and with :math:`\\mathbf{1}` the vector of all ones,
    we have :math:`tc= e^{tA} \\mathbf{1}`.

    Parameters
    __________
    G: Graph or DiGraph object

        a graph.
    t: scalar, optional

        when exponentiating multiply the adjacency matrix by :math:`t`, default 1.


    :return: **tc** (dict)  total communicability of all the nodes of :math:`G`.

    Examples
    ________
    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`.

    .. code:: python

     >>>    G = nx.Graph()
     >>>    G.add_edge(1, 2)
     >>>    G.add_edge(2, 3)
            EdgeView([(1, 2), (2, 3)])

    Compute :math:`tc= e^A \\mathbf{1}`.

     >>>    tc = cm.total_communicability(G)


    References
    ----------
    .. [1] Michele Benzi, Christine Klymko (2013)
           Total communicability as a centrality measure,
           Journal of Complex Networks, Volume 1, Issue 2, Pages 124–149,
           https://doi.org/10.1093/comnet/cnt007
    """

    # G = convert_graph_formats(G,nx.graph)
    n = G.number_of_nodes()
    node_list = G.nodes
    e = ones(n)  # vector of all ones
    Adj = nx.adjacency_matrix(G)
    if t != 1:
        Adj = Adj*t
    tot_communicability = expm_multiply(Adj, e)
    centrality = dict(zip(node_list, tot_communicability))
    return centrality


def node_total_communicability(G, u, t=1, tol=1e-7, maxit=50):

    """
    Computes the node total communicability of node :math:`u`.

    If node :math:`u` is the :math:`i^{th}` node of the graph, the node total communicability of :math:`u` is given by :math:`\\sum_{j=1}^n (e^{tA})_{ij}`,
    where :math:`A` denotes the adjacency matrix of the graph [1]_.

    Parameters
    __________

    G: Graph object
        a graph.
    u: node_id
        node in G.
    t: scalar, optional
     when exponentiating multiply the adjacency matrix by t, default: 1.
    tol: float,optional
     tolerance for convergence, relative accuracy, default: 1e-7.
    maxit: integer, optional
     maximum number of Lanczos iterations, default: 50.


    :return: **tc_u** (float) node total communicability of :math:`u`.

    Examples
    ________
    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`.

    .. code:: python

     >>>    G = nx.Graph()
     >>>    G.add_edge(1, 'u')
     >>>    G.add_edge('u', 2)
            EdgeView([(1, 'u'), ('u', 2)])

    Compute the node total communicability

     >>>    tc_u = cm.node_total_communicability(G, 'u')

    References
    ----------
    .. [1] Michele Benzi, Christine Klymko (2013)
           Total communicability as a centrality measure,
           Journal of Complex Networks, Volume 1, Issue 2, Pages 124–149,
           https://doi.org/10.1093/comnet/cnt007
    """

    n = G.number_of_nodes()
    node_list = G.nodes
    enumerated_nodes = dict(zip(node_list, arange(n)))
    node_position = enumerated_nodes[u]
    e_node = zeros(n)
    e_node[node_position] = 1
    Adj = nx.adjacency_matrix(G)
    if t != 1:
        Adj = Adj*t
    e = ones(n)
    node_communicability = exponential_quadrature(Adj, e, e_node, tol, maxit)
    return node_communicability


def total_network_communicability(G, t=1, tol=1e-7, maxit=50, normalized=False):

    """
    Computes the total network communicability of :math:`G`.

    Total network communicability is defined as the sum over all elements of the exponential of the adjacency matrix\
     [1]_, so denoting with
    :math:`A` the adjacency matrix of graph :math:`G` and with :math:`\\mathbf{1}` the vector of all ones,
    we have :math:`tnc = \\mathbf{1}^T e^{tA} \\mathbf{1}`.

    Sometimes it could be useful to normalize the total network communicability by the number :math:`n` of nodes in the graph to obtain the average total communicability of the network per node: :math:`\\frac{\\mathbf{1}^T e^{tA} \\mathbf{1}}{n}`; this can be done by setting ``normalized = True``.

    Parameters
    __________
    G: Graph object
        an undirected graph.
    t: scalar, optional
        exponentiate multiply the adjacency matrix by t, default: 1.
    tol: float,optional
        tolerance for convergence, relative accuracy, default: 1e-7.
    maxit: integer, optional
        maximum number of Lanczos iterations, default: 50.
    normalized: boolean
        If ``True`` divide the total network communicability by the number of nodes of :math:`G`. Default: ``False``.


    :return: **tnc** (float)  total network communicability of graph :math:`G`.

    Examples
    ________
    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`.

    .. code:: python

     >>>    G = nx.Graph()
     >>>    G.add_edge(1, 'u')
     >>>    G.add_edge('u', 2)
            EdgeView([(1, 'u'), ('u', 2)])

    Compute the total network communicability.

     >>>    tnc = cm.total_network_communicability(G)

    References
    ----------
    .. [1] Michele Benzi, Christine Klymko (2013)
           Total communicability as a centrality measure,
           Journal of Complex Networks, Volume 1, Issue 2, Pages 124–149,
           https://doi.org/10.1093/comnet/cnt007
    """

    n = G.number_of_nodes()
    Adj = nx.adjacency_matrix(G)
    if t != 1:
        Adj = Adj*t
    e = ones(n)
    tot_net_communicability = exponential_symmetric_quadrature(Adj, e, tol, maxit)
    if normalized:
        tot_net_communicability = tot_net_communicability/n
    return tot_net_communicability


def edge_total_communicability(G, u, v, t=1, tol=1e-7, maxit=50):
    """
    Computes the edge total communicabilities of edge :math:`(u, v)`.

    If nodes :math:`u` and :math:`v` are the :math:`i^{th}` and :math:`j^{th}` nodes of the graph, the edge total communicability of :math:`(u, v)` is given by the product of their \
    node total communicability, :math:`(\\sum_{k=1}^n (e^A)_{ik})(\\sum_{k=1}^n (e^A)_{jk})`,
    where :math:`A` denotes the adjacency matrix of the graph [1]_.

    Parameters
    __________

    G: Graph object
        a graph.
    u: node_id
        node in :math:`G`.
    v: node_id
        node in :math:`G`.
    t: scalar, optional
     when exponentiating multiply the adjacency matrix by t, default: 1.
    tol: float,optional
     tolerance for convergence, relative accuracy, default: 1e-7.
    maxit: integer, optional
     maximum number of Lanczos iterations, default: 50


    :return: **tc** (float) total communicability of edge :math:`(u,v)`.

    Examples
    ________
    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`

    .. code:: python

     >>>    G = nx.Graph()
     >>>    G.add_edge(1, 2)
     >>>    G.add_edge(2, 3)
            EdgeView([(1, 2), (2, 3)])

    Compute the edge total communicability of edge :math:`(1,2)`.

     >>>    tc_12 = cm.edge_total_communicability(G, 1, 2)

    References
    ----------
    .. [1] Michele Benzi, Francesca Arrigo (2015)
           Edge Modification Criteria for Enhancing the Communicability of Digraphs,
           SIAM Journal on Matrix Analysis and Applications 37(1),
           https://doi.org/10.1137/15M1034131
    """

    n = G.number_of_nodes()
    node_list = G.nodes
    enumerated_nodes = dict(zip(node_list, arange(n)))
    node_position_1 = enumerated_nodes[u]
    node_position_2 = enumerated_nodes[v]
    e_node_1 = zeros(n)
    e_node_1[node_position_1] = 1
    e_node_2 = zeros(n)
    e_node_2[node_position_2] = 1
    Adj = nx.adjacency_matrix(G)
    if t != 1:
        Adj = Adj * t
    e = ones(n)
    node_communicability_1 = exponential_quadrature(Adj, e, e_node_1, tol, maxit)
    node_communicability_2 = exponential_quadrature(Adj, e, e_node_2, tol, maxit)
    edge_communicability = node_communicability_1*node_communicability_2
    return edge_communicability


# =============================
# DIRECTED NETWORK FUNCTIONS
# =============================


def directed_subgraph_centrality(G, t=1, tol=1e-7, maxit=50):

    """
    Computes the hub and the authority centrality of all nodes in a directed graph :math:`G`.

    Denoting with :math:`A` the adjacency matrix of :math:`G`, with :math:`\\mathcal{A}=\\begin{pmatrix} 0 & A \\\\ A^T & 0 \\end{pmatrix}` the adjacency matrix of the associated undirected bipartite graph, with :math:`\\mathbf{0}` and :math:`e_i` the vector of all zeros and the :math:`i^{th}` vector of the canonical basis , hub centrality and authority centrality of the :math:`i^{th}` node of :math:`G` are defined as

     :math:`\\phantom{aaaaaaa} h_i(G) = [e^\\mathcal{A}]_{ii} = e_i^T\\cosh{\\left(\\sqrt{A A^T}\\right)}e_i  = \\begin{pmatrix} e_i^T & \\mathbf{0}^T \\end{pmatrix}  e^{\\mathcal{A}}\\begin{pmatrix} e_i \\\\ \\mathbf{0} \\end{pmatrix}`,
     :math:`\\phantom{aaaaa}a_i(G) = [e^\\mathcal{A}]_{N+i N+i} = e_i^T\\cosh{\\left(\\sqrt{A^T A}\\right)}e_i = \\begin{pmatrix} \\mathbf{0}^T & e_i^T \\end{pmatrix}  e^{\\mathcal{A}}\\begin{pmatrix} \\mathbf{0} \\\\ e_i \\end{pmatrix}`.

    See [1]_ for further details.

    Parameters
    __________

    G: DiGraph object
        a directed graph.
    t: scalar, optional
     when computing the total hub and authority communicabilities multiply the adjacency matrix by :math:`t`, default: 1.
    tol: float,optional
     tolerance for convergence, relative accuracy, default: 1e-7.
    maxit: integer, optional
     maximum number of Lanczos iterations, default: 50

    Returns
    ________

    hc: dict
     hub centrality.
    ac: dict
     authority centrality.

    Examples
    ________

    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`.

    .. code:: python

     >>>    G = nx.DiGraph()
     >>>    G.add_edge(1, 2)
     >>>    G.add_edge(1, 3)
     >>>    G.add_edge(2, 3)
     >>>    G.add_edge(3, 1)
            OutEdgeView([(1, 2), (1, 3), (2, 3), (3, 1)])

    Compute hub and authority centrality.

     >>>    hc, ac = cm.directed_subgraph_centrality(G)

    References
    ----------
    .. [1] Michele Benzi, Ernesto Estrada and Christine Klymko (2013),
           Ranking hubs and authorities using matrix functions,
           Linear Algebra Appl., 438, 2447–2474.
           https://doi.org/10.1016/j.laa.2012.10.022
    """
    n = G.number_of_nodes()
    node_list = list(G.nodes)
    Adj = nx.adjacency_matrix(G)
    Bip_Adj = bmat([[None, Adj], [Adj.transpose(), None]])
    if t != 1:
        Bip_Adj = Bip_Adj * t
    hub_centrality = np.zeros(n)
    authority_centrality = np.zeros(n)
    for i in range(n):
        h_node = zeros(2*n)
        h_node[i] = 1
        a_node = zeros(2*n)
        a_node[n+i] = 1
        hub_centrality[i] = exponential_symmetric_quadrature(Bip_Adj, h_node, tol, maxit)
        authority_centrality[i] = exponential_symmetric_quadrature(Bip_Adj, a_node, tol, maxit)
    hub_centrality = dict(zip(node_list, hub_centrality))
    authority_centrality = dict(zip(node_list, authority_centrality))
    return hub_centrality, authority_centrality


def node_directed_subgraph_centrality(G, u, t=1, tol=1e-7, maxit=50):
    """
    Computes the hub and the authority centrality of node :math:`u`.

    If node :math:`u` is the :math:`i^{th}` node of the graph, denoting with :math:`A` the adjacency matrix of :math:`G`, with :math:`\\mathcal{A}=\\begin{pmatrix} 0 & A \\\\ A^T & 0 \\end{pmatrix}` the adjacency matrix of the associated undirected bipartite graph, with :math:`\\mathbf{0}` and :math:`e_i` the vector of all zeros and the :math:`i^{th}` vector of the canonical basis , hub centrality and authority centrality of the :math:`i^{th}` node of :math:`G` are defined as

     :math:`\\phantom{aaaaaaa} h_i(G) = [e^\\mathcal{A}]_{ii} = e_i^T\\cosh{\\left(\\sqrt{A A^T}\\right)}e_i  = \\begin{pmatrix} e_i^T & \\mathbf{0}^T \\end{pmatrix}  e^{\\mathcal{A}}\\begin{pmatrix} e_i \\\\ \\mathbf{0} \\end{pmatrix}`,
     :math:`\\phantom{aaaaa}a_i(G) = [e^\\mathcal{A}]_{N+i N+i} = e_i^T\\cosh{\\left(\\sqrt{A^T A}\\right)}e_i = \\begin{pmatrix} \\mathbf{0}^T & e_i^T \\end{pmatrix}  e^{\\mathcal{A}}\\begin{pmatrix} \\mathbf{0} \\\\ e_i \\end{pmatrix}`.

    See [1]_ for further details.

    Parameters
    __________

    G: DiGraph object
        a directed graph.
    u: node_id
        node in :math:`G`.
    t: scalar, optional
     when computing the total hub and authority communicabilities multiply the adjacency matrix by :math:`t`, default: 1.
    tol: float,optional
     tolerance for convergence, relative accuracy, default: 1e-7.
    maxit: integer, optional
     maximum number of Lanczos iterations, default: 50

    Returns
    ________

    hc_u: dict
     hub centrality of :math:`u`..
    ac_u: dict
     authority centrality of :math:`u`..

    Examples
    ________

    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`.

    .. code:: python

     >>>    G = nx.DiGraph()
     >>>    G.add_edge(1, 2)
     >>>    G.add_edge(1, 3)
     >>>    G.add_edge(2, 3)
     >>>    G.add_edge(3, 1)
            OutEdgeView([(1, 2), (1, 3), (2, 3), (3, 1)])

    Compute hub and authority centrality.

     >>>    hc, ac = cm.directed_subgraph_centrality(G)

    References
    ----------
    .. [1] Michele Benzi, Ernesto Estrada and Christine Klymko (2013),
           Ranking hubs and authorities using matrix functions,
           Linear Algebra Appl., 438, 2447–2474.
           https://doi.org/10.1016/j.laa.2012.10.022
    """

    n = G.number_of_nodes()
    node_list = list(G.nodes)
    enumerated_nodes = dict(zip(node_list, arange(n)))
    node_position = enumerated_nodes[u]
    Adj = nx.adjacency_matrix(G)
    Bip_Adj = bmat([[None, Adj], [Adj.transpose(), None]])
    if t != 1:
        Bip_Adj = Bip_Adj * t
    h_node = zeros(2 * n)
    h_node[node_position] = 1
    a_node = zeros(2 * n)
    a_node[n + node_position] = 1
    hub_centrality = exponential_symmetric_quadrature(Bip_Adj, h_node, tol, maxit)
    authority_centrality  = exponential_symmetric_quadrature(Bip_Adj, a_node, tol, maxit)
    return hub_centrality, authority_centrality


def total_directed_communicability(G, t=1):

    """
    Computes the total hub communicability and the total authority communicability of a directed graph :math:`G`.

    Denoting with :math:`A` the adjacency matrix of :math:`G`, with :math:`\\mathcal{A}=\\begin{pmatrix} 0 & A \\\\ A^T & 0 \\end{pmatrix}` the adjacency matrix of the associated undirected bipartite graph and with :math:`\\mathbf{0}` and :math:`\mathbf{1}` the vectors of all zeros and ones respectively, total hub communicability and total authority communicability  of :math:`G` are defined as

     :math:`T_{h}C(G) = \\mathbf{1}^T\\cosh{\\left(\\sqrt{A A^T}\\right)}\\mathbf{1} = \\begin{pmatrix} \\mathbf{1}^T & \\mathbf{0}^T \\end{pmatrix}  e^{\\mathcal{A}}\\begin{pmatrix} \\mathbf{1} \\\\ \\mathbf{0} \\end{pmatrix}`,
     :math:`T_{a}C(G) = \\mathbf{1}^T\\cosh{\\left(\\sqrt{A^T A}\\right)}\\mathbf{1} = \\begin{pmatrix} \\mathbf{0}^T & \\mathbf{1}^T \\end{pmatrix}  e^{\\mathcal{A}}\\begin{pmatrix} \\mathbf{0} \\\\ \\mathbf{1} \\end{pmatrix}`.

    See [1]_ for further details.

    Parameters
    __________

    G: DiGraph object
        a directed graph.
    t: scalar, optional
     when computing the total hub and authority communicabilities multiply the adjacency matrix by :math:`t`, default: 1.

    Returns
    ________

    thc: float
     total hub communicability.
    trc: float
     total authority communicability.

    Examples
    ________

    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`.

    .. code:: python

     >>>    G = nx.DiGraph()
     >>>    G.add_edge(1, 2)
     >>>    G.add_edge(1, 3)
     >>>    G.add_edge(2, 3)
     >>>    G.add_edge(3, 1)
            OutEdgeView([(1, 2), (1, 3), (2, 3), (3, 1)])

    Compute total hub communicability and total authority communicability.

     >>>    thc, tac = cm.total_directed_communicability(G)

    References
    ----------
    .. [1] Benzi M. & Arrigo F. (2016),
           Edge Modification Criteria for Enhancing the Communicability of Digraphs,
           SIAM J. Matrix Anal. Appl., 37(1), 443–468.
           https://doi.org/10.1137/15M1034131
    """

    n = G.number_of_nodes()
    node_list = G.nodes
    one_zero = zeros(2*n)
    one_zero[:n] = ones(n)
    zero_one = zeros(2*n)
    zero_one[n:] = ones(n)
    Adj = nx.adjacency_matrix(G)
    Bip_Adj = bmat([[None, Adj], [Adj.transpose(), None]])
    if t != 1:
        Bip_Adj = Bip_Adj*t
    tot_hub_communicability = expm_multiply(Bip_Adj, one_zero)
    thc = np.sum(tot_hub_communicability[:n])
    tot_authority_communicability = expm_multiply(Bip_Adj, zero_one)
    tac = np.sum(tot_authority_communicability[n:])
    return thc, tac


def node_total_directed_communicability(G, u, t=1, tol=1e-7, maxit=50):

    """
    Computes the total hub and authority communicability of node :math:`u`.

    If node :math:`u` is the :math:`i^{th}` node of the graph, denoting with :math:`A` the adjacency matrix of :math:`G`, with :math:`\\mathcal{A}=\\begin{pmatrix} 0 & A \\\\ A^T & 0 \\end{pmatrix}` the adjacency matrix of the associated undirected bipartite graph, with :math:`\\sinh^{\\diamondsuit}` the generalized hyperbolic sine and with :math:`\\mathbf{0}`, :math:`\\mathbf{1}`, :math:`\\mathbf{e_i}`  the vectors of all zeros, of all ones and of all zeros except for a :math:`1` in position :math:`i` respectively, the total hub communicability of :math:`u` and the total authority communicability of :math:`u` are defined as

     :math:`T_{h}C(u) = \\mathbf{e_i}^T\\sinh^{\\diamondsuit}(A)\\mathbf{1} = \\begin{pmatrix} \\mathbf{e_i}^T & \\mathbf{0}^T \\end{pmatrix}  e^{\\mathcal{A}}\\begin{pmatrix} \\mathbf{0} \\\\ \\mathbf{1} \\end{pmatrix},`
     :math:`T_{a}C(u) = \\mathbf{e_i}^T\\sinh^{\\diamondsuit}(A)^T\\mathbf{1} = \\begin{pmatrix} \\mathbf{0}^T & \\mathbf{e_i}^T \\end{pmatrix}  e^{\\mathcal{A}}\\begin{pmatrix} \\mathbf{1} \\\\ \\mathbf{0} \\end{pmatrix}.`

    See [1]_ for further details.


    Parameters
    __________

    G: DiGraph object
        a directed graph.
    u: node_id
        node in :math:`G`.
    t: scalar, optional
     when computing the total hub and authority communicabilities of :math:`u` multiply the adjacency matrix by :math:`t`, default: 1.
    tol: float,optional
     tolerance for convergence, relative accuracy; default: 1e-7.
    maxit: integer, optional
     maximum number of Lanczos iterations; default: 50.


    Returns
    __________
     thc: float

        total hub communicability of :math:`u`.

    tac: float

        total authority communicability of :math:`u`.


    Examples
    ________

    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import networkx as nx

    Create graph :math:`G`

    .. code:: python

     >>>    G = nx.DiGraph()
     >>>    G.add_edge(1, 2)
     >>>    G.add_edge(1, 3)
     >>>    G.add_edge(2, 3)
     >>>    G.add_edge(3, 1)
            OutEdgeView([(1, 2), (1, 3), (2, 3), (3, 1)])

    Compute total hub and authority communicabilities of node 1

     >>>    thc, tac = cm.node_total_directed_communicability(G, 1)

    References
    ----------
    .. [1] Benzi M. & Arrigo F. (2016),
            Edge Modification Criteria for Enhancing the Communicability of Digraphs,
           SIAM J. Matrix Anal. Appl., 37(1), 443–468.
           https://doi.org/10.1137/15M1034131
    """

    n = G.number_of_nodes()
    node_list = G.nodes
    enumerated_nodes = dict(zip(node_list, arange(n)))
    node_position = enumerated_nodes[u]
    e_node = zeros(n)
    e_node[node_position] = 1
    Adj = nx.adjacency_matrix(G)
    Bip_Adj = bmat([[None, Adj], [Adj.transpose(), None]])
    e_node_zero = zeros(2*n)
    e_node_zero[:n] = e_node
    zero_one = zeros(2*n)
    zero_one[n:] = ones(n)
    zero_e_node = zeros(2*n)
    zero_e_node[n:] = e_node
    one_zero = zeros(2*n)
    one_zero[:n] = ones(n)
    if t != 1:
        Bip_Adj = Bip_Adj*t
    thc = exponential_quadrature(Bip_Adj, e_node_zero, zero_one, tol, maxit)
    tac = exponential_quadrature(Bip_Adj, zero_e_node, one_zero, tol, maxit)
    return thc, tac


# =============================
# TEMPORAL CENTRALITY MEASURES
# =============================

def broadcast_centrality(G, alpha=None, conj_grad_maxiter=100, conj_grad_tol=1e-7):
    """
    Computes the broadcast centrality of the dynamic graph :math:`G`.

    Denoting with :math:`A_t` the adjacency matrix at time :math:`t` and with :math:`\\mathbf{1}` the vector of all ones
    broadcast centrality is :math:`bc = (I-\\alpha A_1)^{-1}(I-\\alpha A_2)^{-1}...(I-\\alpha A_k)^{-1} \\mathbf{1}`\
     [1]_.

    Recall that to ensure that each resolvent :math:`(I-\\alpha A_t)` can be expressed as a power series in the matrix,
    the parameter :math:`\\alpha` must satisfy\
     :math:`0<\\alpha< \\frac{1}{\\rho^*}`, where :math:`\\rho^* = \\max_t \\rho(A_t)`.

    Parameters
    __________

    G: DynGraph object
        a dynamic graph.
    alpha: float, optional
        parameter, if None computed to ensure that each resolvent can be expressed as a power series in the matrix, default: None.
    conj_grad_maxiter: integer
        Maximum number of iterations for solving :math:`Ax=b` with conjugate gradient method. \
        Iterations will stop after maxiter steps even if the specified tolerance has not been achieved, default: 100.
    conj_grad_tol: float, optional
        Tolerance for solving :math:`Ax=b` with conjugate gradient method.
         ``norm(residual) <= conj_grad_tol*norm(b)`` , default: 1e-7.


    Returns
    _______

    bc: dict
        broadcast centrality.

    alpha: float
        alpha parameter.

    Examples
    ________

    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import dynetx as dn

    Create dynamic graph :math:`G`

    .. code:: python

     >>>    G = dn.DynGraph()
     >>>    G.add_interaction(1, 2, 2, 5)
     >>>    G.add_interaction(1, 3, 2, 5)
     >>>    G.add_interaction(2, 3, 4)
             EdgeView([(1, 2), (1, 3), (2, 3)])

    Compute broadcast centrality

     >>>   bc, alpha = cm.broadcast_centrality(G)


    References
    ----------
    .. [1] Michele Benzi, Isabel Chen, Howard H. Chang, Vicki S. Hertzberg (2017),
           Dynamic communicability and epidemic spread: a case study on an empirical dynamic contact network,
           Journal of Complex Networks, Volume 5, Issue 2,
           https://doi.org/10.1093/comnet/cnw017
    """

    time_snapshots = G.temporal_snapshots_ids()
    n = G.number_of_nodes()
    e = ones(n)  # vector of all ones
    bc = e
    if alpha is None:
        spectral_bound = 0
        for t in time_snapshots[::-1]:
            G_t = graph_slice(G, t)
            Adj_t = nx.adjacency_matrix(G_t)
            row_sum = scipy_norm(Adj_t, ord=inf)
            spectral_bound = max(spectral_bound, row_sum)
        if spectral_bound != 0:
            alpha = 0.9/spectral_bound  # 0 < alpha < 0.9/max(rho(Adj_t))
    for t in time_snapshots[::-1]:
        G_t = graph_slice(G, t)
        Adj_t = nx.adjacency_matrix(G_t)
        bc = cg(identity(n) - alpha * Adj_t, bc, maxiter=conj_grad_maxiter, tol=conj_grad_tol, atol=0)
        if bc[1] != 0:
            raise ValueError('convergence not achieved')
        bc = bc[0]
        if any(bc != 0):
            bc = bc / norm(bc)
    G_0 = graph_slice(G, time_snapshots[0])
    node_list = list(G_0.nodes)
    bc = dict(zip(node_list, bc))
    return bc, alpha


def receive_centrality(G, alpha=None, conj_grad_maxiter=100, conj_grad_tol=1e-7):
    """
            Computes the receive centrality of the dynamic graph :math:`G`.

            Denoting with :math:`A_t` the adjacency matrix at time :math:`t` and with :math:`\\mathbf{1}` the vector of all ones
            receive centrality is :math:`rc = (I-\\alpha A_k)^{-1}(I-\\alpha A_{k-1})^{-1}...(I-\\alpha A_1)^{-1} \\mathbf{1}`\
             [1]_.

            Recall that to ensure that each resolvent :math:`(I-\\alpha A_t)` can be expressed as a power series in the matrix,
            the parameter :math:`\\alpha` must satisfy \
            :math:`0<\\alpha< \\frac{1}{\\rho^*}`, where :math:`\\rho^* = \\max_t \\rho(A_t)`.

    Parameters
    __________

    G: DynGraph object
        a dynamic graph.
    alpha: float, optional
        parameter, if None computed to ensure that each resolvent can be expressed as a power series in the matrix. Default: None.
    conj_grad_maxiter: integer
        Maximum number of iterations for solving :math:`Ax=b` with conjugate gradient method.\
         Iterations will stop after maxiter steps even if the specified tolerance has not been achieved, default: 100.
    conj_grad_tol: float, optional
        Tolerance for solving :math:`Ax=b` with conjugate gradient method.
         ``norm(residual) <= conj_grad_tol*norm(b)`` , default: 1e-7.


    Returns
    _______

    rc: dict

        receive centrality.

    alpha: float

        alpha parameter.

    Examples
    ________

    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import dynetx as dn

    Create dynamic graph :math:`G`.

    .. code:: python

     >>>    G = dn.DynGraph()
     >>>    G.add_interaction(1, 2, 2, 5)
     >>>    G.add_interaction(1, 3, 2, 5)
     >>>    G.add_interaction(2, 3, 4)
             EdgeView([(1, 2), (1, 3), (2, 3)])

    Compute receive centrality

     >>>   rc, alpha = cm.receive_centrality(G)


    References
    ----------
    .. [1] Michele Benzi, Isabel Chen, Howard H. Chang, Vicki S. Hertzberg (2017),
           Dynamic communicability and epidemic spread: a case study on an empirical dynamic contact network,
           Journal of Complex Networks, Volume 5, Issue 2,
           https://doi.org/10.1093/comnet/cnw017
    """

    time_snapshots = G.temporal_snapshots_ids()
    n = G.number_of_nodes()
    e = ones(n)  # vector of all ones
    rc = e
    if alpha is None:
        spectral_bound = 0
        for t in time_snapshots:
            G_t = graph_slice(G, t)
            Adj_t = nx.adjacency_matrix(G_t)
            row_sum = scipy_norm(Adj_t, ord=inf)
            spectral_bound = max(spectral_bound, row_sum)
        if spectral_bound != 0:
            alpha = 0.9/spectral_bound  # 0 < alpha < 0.9/max(rho(Adj_t))
    for t in time_snapshots:
        G_t = graph_slice(G, t)
        Adj_t = nx.adjacency_matrix(G_t)
        rc = cg(identity(n) - alpha * Adj_t, rc, maxiter=conj_grad_maxiter, tol=conj_grad_tol, atol=0)
        if rc[1] != 0:
            raise ValueError('convergence not achieved')
        rc = rc[0]
        if any(rc != 0):
            rc = rc / norm(rc)
    G_0 = graph_slice(G, time_snapshots[0])
    node_list = list(G_0.nodes)
    rc = dict(zip(node_list, rc))
    return rc, alpha


def approximated_broadcast_centrality(G, alpha=None):
    """
    Computes an approximated version of the broadcast centrality.

    Denoting with :math:`A_t` the adjacency matrix at time :math:`t` and with :math:`\\mathbf{1}` the vector of all ones
    approximated broadcast centrality is given by \
    :math:`bc = (I+\\alpha A_1)(I+\\alpha A_2)...(I+\\alpha A_k)^{-1} \\mathbf{1}` [1]_.

    This formula is an approximated version of the classic broadcast centrality \
    :math:`bc = (I-\\alpha A_1)^{-1}(I-\\alpha A_2)^{-1}...(I-\\alpha A_k)^{-1} \\mathbf{1}`
    where each power series is truncated to the first order.


    Parameters
    __________

    G: DynGraph object
        a dynamic graph.
    alpha: float, optional
        parameter, if None :math:`\\alpha = 0.9 \\frac{1}{\\rho^*}`, where :math:`\\rho^* = \\max_t \\rho(A_t)`,\
         default None.


    Returns
    _______

     bc: dict

        approximated broadcast centrality.

    alpha: float

        alpha parameter.

    Examples
    ________

    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import dynetx as dn

    Create dynamic graph :math:`G`

    .. code:: python

     >>>    G = dn.DynGraph()
     >>>    G.add_interaction(1, 2, 2, 5)
     >>>    G.add_interaction(1, 3, 2, 5)
     >>>    G.add_interaction(2, 3, 4)
             EdgeView([(1, 2), (1, 3), (2, 3)])

    Compute approximated broadcast centrality

     >>>   bc, alpha = cm.approximated_broadcast_centrality(G)

    References
    ----------
    .. [1] Michele Benzi, Isabel Chen, Howard H. Chang, Vicki S. Hertzberg (2017),
           Dynamic communicability and epidemic spread: a case study on an empirical dynamic contact network,
           Journal of Complex Networks, Volume 5, Issue 2,
           https://doi.org/10.1093/comnet/cnw017
    """

    time_snapshots = G.temporal_snapshots_ids()
    n = G.number_of_nodes()
    e = ones(n)  # vector of all ones
    bc = e
    return_alpha = False
    if alpha is None:
        return_alpha = True
        spectral_bound = 0
        for t in time_snapshots:
            G_t = graph_slice(G, t)
            Adj_t = nx.adjacency_matrix(G_t)
            row_sum = scipy_norm(Adj_t, ord=inf)
            spectral_bound = max(spectral_bound, row_sum)
        if spectral_bound != 0:
            alpha = 0.9 / spectral_bound  # 0 < alpha < 0.9/max(rho(Adj_t))
    for t in time_snapshots[::-1]:
        G_t = graph_slice(G, t)
        Adj_t = nx.adjacency_matrix(G_t)
        bc = bc + alpha * Adj_t.dot(bc)
        if any(bc != 0):
            bc = bc / norm(bc)
    G_0 = graph_slice(G, time_snapshots[0])
    node_list = list(G_0.nodes)
    bc = dict(zip(node_list, bc))
    if return_alpha:
        return bc, alpha
    else:
        return bc


def approximated_receive_centrality(G, alpha=None):
    """
    Computes an approximated version of the receive centrality.

    Denoting with :math:`A_t` the adjacency matrix at time :math:`t` and with :math:`\\mathbf{1}` the vector of all ones
    approximated receive centrality is given by\
     :math:`rc = (I+\\alpha A_k)(I+\\alpha A_{k-1})...(I+\\alpha A_1)^{-1} \\mathbf{1}` [1]_.

    This formula is an approximated version of the classic receive centrality\
     :math:`rc = (I-\\alpha A_k)^{-1}(I-\\alpha A_{k-1})^{-1}...(I-\\alpha A_1)^{-1} \\mathbf{1}`
    where each power series is truncated to the first order.


    Parameters
    __________

    G: DynGraph object
        a dynamic graph.
    alpha: float, optional
        parameter, if None :math:`\\alpha = 0.9 \\frac{1}{\\rho^*}`, where :math:`\\rho^* = \\max_t \\rho(A_t)`,\
         default None.

    Examples
    ________

    .. code:: python

     >>>  from networksns import centrality_measures as cm
     >>>  import dynetx as dn

    Create dynamic graph :math:`G`

    .. code:: python

     >>>    G = dn.DynGraph()
     >>>    G.add_interaction(1, 2, 2, 5)
     >>>    G.add_interaction(1, 3, 2, 5)
     >>>    G.add_interaction(2, 3, 4)
             EdgeView([(1, 2), (1, 3), (2, 3)])

    Compute approximated receive centrality

     >>>   bc, alpha = cm.approximated_receive_centrality(G)

    Returns
    _______

     rc: dict

        approximated receive centrality.

    alpha: float

        alpha parameter.

    References
    ----------
    .. [1] Michele Benzi, Isabel Chen, Howard H. Chang, Vicki S. Hertzberg (2017),
           Dynamic communicability and epidemic spread: a case study on an empirical dynamic contact network,
           Journal of Complex Networks, Volume 5, Issue 2,
           https://doi.org/10.1093/comnet/cnw017
    """

    time_snapshots = G.temporal_snapshots_ids()
    n = G.number_of_nodes()
    e = ones(n)  # vector of all ones
    rc = e
    return_alpha = False
    if alpha is None:
        return_alpha = True
        spectral_bound = 0
        for t in time_snapshots:
            G_t = graph_slice(G, t)
            Adj_t = nx.adjacency_matrix(G_t)
            row_sum = scipy_norm(Adj_t, ord=inf)
            spectral_bound = max(spectral_bound, row_sum)
        if spectral_bound != 0:
            alpha = 0.9 / spectral_bound  # 0 < alpha < 0.9/max(rho(Adj_t))
    for t in time_snapshots:
        G_t = graph_slice(G, t)
        Adj_t = nx.adjacency_matrix(G_t)
        rc = rc + alpha * Adj_t.dot(rc)
        if any(rc != 0):
            rc = rc / norm(rc)
    G_0 = graph_slice(G, time_snapshots[0])
    node_list = list(G_0.nodes)
    rc = dict(zip(node_list, rc))
    if return_alpha:
        return rc, alpha
    else:
        return rc


def trip_centrality(edge_list, alpha, epsilon=None):
    """
        Computes the trip centrality of a temporal multiplex.

        Trip centrality is a generalization of Katz centrality to the case of a temporal multiplex with non-zero link
        travel time. It counts the paths that can be travelled according to the network temporal structure while
        also differentiating the contributions of inter- and intra-layer walks to centrality.

        The parameters :math:`\\alpha` and :math:`\\epsilon` determine the value of the contribution given by a walk of
        length :math:`n` changing layer :math:`m` times. This value is :math:`{\\alpha}^n{\\varepsilon}^m`.

        For a full description of the algorithm see [1]_.


        Parameters
        __________

        edge_list: List of lists
            When there are multiple layers the edge structure is the following
            ``[departure_node, arrival_node, departure_time, arrival_time, layer]``.
            While with only one layer
            ``[departure_node, arrival_node, departure_time, arrival_time]``.
        alpha: float, optional
            parameter.
        epsilon: float, optional
            parameter, if None changing layer does not influence the contribution of a walk.
            When there is only one layer does not need to be specified, default: None.


        :return: **tc**: (dictionary)
            trip centrality.


        Examples
        ________

        .. code:: python

         >>>  from networksns import centrality_measures as cm         >>>           >>>

        Create multiplex edge list.

        .. code:: python

         >>>    edge_list = [['u', 'v' ,1 , 3, 'l'], ['u', 'w', 1, 2, 'l'], ['w', 'u', 1, 5, 'm'],
         >>>                 ['u', 'v' ,2 , 4, 'm'], ['u', 'w', 3, 4, 'm'], ['w', 'u', 4, 5, 'r'],
         >>>                 ['u', 'v' ,1 , 5, 'm'], ['v', 'w', 2, 3, 'm'], ['v', 'u', 4, 5, 'r']]

        Compute trip centrality.

         .. code:: python

        >>>   tc = cm.trip_centrality(edge_list, 0.3, 0.2)

        References
        ----------
        .. [1] Silvia Zaoli, Piero Mazzarisi & Fabrizio Lillo (2019),
               Trip Centrality: walking on a temporal multiplex with non-instantaneous link travel time.
               Scientific Reports, 9, Article number: 10570.
               https://www.nature.com/articles/s41598-019-47115-6
        """

    tc = 0
    alpha_tilde = alpha ** (1 / 2)
    if len(edge_list[0]) == 4:
        # only 1 layer
        primary_nodes = []
        times = []
        n_secondary = len(edge_list)
        rows = []
        columns = []
        count = 2 * n_secondary
        for e in edge_list:
            if e[0] not in primary_nodes:
                primary_nodes.append(e[0])
            if e[1] not in primary_nodes:
                primary_nodes.append(e[1])
            if e[2] not in times:
                bisect.insort(times, e[2])
                rows.append([])
                columns.append([])
            if e[3] not in times:
                bisect.insort(times, e[3])
                rows.append([])
                columns.append([])
            rows[times.index(e[2])].append(primary_nodes.index(e[0]))
            columns[times.index(e[2])].append(count)
            rows[times.index(e[3])].append(count)
            columns[times.index(e[3])].append(primary_nodes.index(e[1]))
            count = count + 1
        n_primary = len(primary_nodes)
        tc = ones(n_primary + n_secondary)
        for t in times:
            A_t = csr_matrix((ones(len(rows[times.index(t)])), (rows[times.index(t)], columns[times.index(t)])),
                             shape=(3 * n_secondary, 3 * n_secondary), dtype=int)
            A12_t = A_t[:n_primary, 2 * n_secondary:]
            A21_t = A_t[2 * n_secondary:, :n_primary]
            A_t = bmat([[None, A21_t], [A12_t, None]], format='csr')
            tc = alpha_tilde * A_t.dot(tc) + tc
        tc = tc - ones(n_primary + n_secondary)
        tc = dict(zip(primary_nodes, tc))
    # -----------------------------------------------
    if len(edge_list[0]) == 5:  # multiple layers
        primary_nodes = []
        times = []
        n_secondary = len(edge_list)
        layers = []
        for e in edge_list:
            primary_nodes.append(e[0])
            primary_nodes.append(e[1])
            times.append(e[2])
            times.append(e[3])
            layers.append(e[4])
        primary_nodes = list(set(primary_nodes))
        n_primary = len(primary_nodes)
        times = list(set(times))
        n_times = len(times)
        layers = list(set(layers))
        n_layers = len(layers)
        n_nodes = n_primary * n_layers + n_secondary
        matrices_data = []
        for i in range(n_times):
            matrices_data.append([[], []])
        edge_count = 0
        for e in edge_list:
            matrices_data[times.index(e[2])][0].append(n_primary * layers.index(e[4]) + primary_nodes.index(e[0]))
            matrices_data[times.index(e[2])][1].append(n_primary * n_layers + edge_count)
            matrices_data[times.index(e[3])][0].append(n_primary * n_layers + edge_count)
            matrices_data[times.index(e[3])][1].append(n_primary * layers.index(e[4]) + primary_nodes.index(e[1]))
            edge_count = edge_count + 1
        if epsilon is not None:
            K_small = ones((n_layers, n_layers)) * epsilon + identity(n_layers) * (1 - epsilon)
            I_primary = identity(n_primary)
            K_big = kron(K_small, I_primary)
            K = bmat([[K_big, None], [None, identity(n_secondary)]], format='csr')
        else:
            K = identity(n_primary*n_layers+n_secondary)
        tc = ones(n_nodes)
        auxiliary = tc
        # solving K^-1*e is equal to divide the first n_primary*n_layers elements by 1+(n_layers-1)*epsilon
        auxiliary[0: n_primary * n_layers] = auxiliary[0: n_primary * n_layers] / (1 + epsilon * (n_layers - 1))
        data = matrices_data[-1]
        # creating last matrix to compute (I+alpha_tilde A)*(K^-1)e
        matrix = csr_matrix((ones(len(data[0])), (data[0], data[1])), shape=(n_nodes, n_nodes), dtype=int)
        tc = auxiliary + alpha_tilde * (matrix.dot(tc))
        for data in matrices_data[n_times - 2::-1]:
            #  (I+alpha_tilde AK)*tc
            matrix = csr_matrix((ones(len(data[0])), (data[0], data[1])), shape=(n_nodes, n_nodes), dtype=int)
            tc = tc + alpha_tilde * matrix.dot(K.dot(tc))
        aggregated_centrality = zeros(n_primary)
        for i in range(n_primary):
            aggregated_centrality = aggregated_centrality + tc[n_primary*i:n_primary*(1+i)]
        tc = dict(zip(primary_nodes, aggregated_centrality))
    return tc


def betweenness_index(n_nodes, n_times, node_index, time_index, layer_index):
    """
    find the index associated to a point in the static graph. See betweenness_centrality.
    """

    index = n_nodes * n_times * layer_index + n_nodes * time_index + node_index
    return index


def inverse_betweenness_index(n_nodes, n_times, index):
    """
    find the triple associated to a point in the static graph. See betweenness_centrality.
    """

    layer_index = index // (n_nodes * n_times)
    time_index = (index % (n_nodes * n_times)) // n_nodes
    node_index = (index % (n_nodes * n_times)) % n_nodes
    return [node_index, time_index, layer_index]


def betweenness_node_family(n_nodes, n_times, n_layers, node_index):
    """
    find the set of nodes in the static graph with node_index as vertex. See betweenness_centrality.
    """
    node_family = range(node_index, n_nodes * n_times * n_layers, n_nodes)

    return node_family


def betweenness_node_time_family(n_nodes, n_times, n_layers, node_index, time_index):
    """
    find the set of nodes in the static graph with node_index as vertex and time_index as starting time.
    See betweenness_centrality.
    """

    node_time_family = range(node_index + n_nodes * time_index, n_nodes * n_times * n_layers, n_nodes * n_times)

    return node_time_family


def betweenness_centrality(edge_list, delta_t, alpha, epsilon):
    """
            Computes the betweenness centrality of a temporal multiplex.

            This centrality is a generalization of betweenness centrality to the case of a temporal multiplex with\
             non-zero link travel time.
             In identifying the shortest paths it takes in account both the temporal and the multiplex\
             structure, differentiating the contributions of inter- and intra-layer walks to centrality.

            The parameters :math:`\\alpha` and :math:`\\epsilon` determine the length of a path according to the\
             formula

            :math:`\\phantom{aaaaaaaaaaaaaaaaaaa}\\mathcal{L} = \\alpha (n+\\varepsilon m) + (1-\\alpha)\\mathcal{T}`.

            where :math:`n` is the number of intra-layer links used, :math:`m`  is the number of inter-layer links,\
             :math:`\\mathcal{T}` is the duration of the path, :math:`\\alpha \\leq 1` and :math:`\\varepsilon \\in\
              [0, +\\infty)`.

            For a full description of the algorithm see [1]_.


            Parameters
            __________

            edge_list: List of lists
                When there are multiple layers the edge structure is the following
                ``[departure_node, arrival_node, departure_time, arrival_time, layer]``.
                While with only one layer
                ``[departure_node, arrival_node, departure_time, arrival_time]``.
            delta_t: int
                Time interval.
            alpha: float, optional
                parameter.
            epsilon: float, optional
                parameter, if None changing layer does not influence the contribution of a walk.
                When there is only one layer does not need to be specified, default: None.


            :return: **bc**: (dictionary)
                betweenness centrality.


            Examples
            ________

            .. code:: python

             >>>  from networksns import centrality_measures as cm             >>>               >>>

            Create multiplex edge list

            .. code:: python

             >>>    edge_list = [['u', 'v' ,1 , 3, 'l'], ['u', 'w', 1, 2, 'l'], ['w', 'u', 1, 5, 'm'],
             >>>                 ['u', 'v' ,2 , 4, 'm'], ['u', 'w', 3, 4, 'm'], ['w', 'u', 4, 5, 'r'],
             >>>                 ['u', 'v' ,1 , 5, 'm'], ['v', 'w', 2, 3, 'm'], ['v', 'u', 4, 5, 'r']]

            Compute betweenness centrality

             .. code:: python

            >>>   bc = cm.betweenness_centrality(edge_list, 1, 0.3, 0.2)

            References
            ----------
            .. [1] Silvia Zaoli, Piero Mazzarisi & Fabrizio Lillo (2021),
                   Betweenness centrality for temporal multiplexes.
                   Scientific Reports volume 11, Article number: 4919.
                   https://www.nature.com/articles/s41598-021-84418-z
            """

    times = []
    nodes = []
    layers = []
    node_layer_time_dict = {}  # departure node: layer: departure time
    for e in edge_list:
        nodes.append(e[0])
        nodes.append(e[1])
        times.append(e[2])
        times.append(e[3])
        layers.append(e[4])
        if e[0] in node_layer_time_dict.keys():
            if e[4] in node_layer_time_dict[e[0]].keys():
                node_layer_time_dict[e[0]][e[4]].append(e[2])
            else:
                node_layer_time_dict[e[0]][e[4]] = [e[2]]
        else:
            node_layer_time_dict[e[0]] = {}
            node_layer_time_dict[e[0]][e[4]] = {}
            node_layer_time_dict[e[0]][e[4]] = [e[2]]

    nodes = list(set(nodes))
    n_nodes = len(nodes)
    t_min = min(set(times))
    t_max = max(set(times))
    times = list(range(t_min, t_max + delta_t, delta_t))
    n_times = len(times)
    layers = list(set(layers))
    n_layers = len(layers)
    static_edges = []
    static_data = []
    static_row = []
    static_column = []
    for e in edge_list:
        node_index_1 = nodes.index(e[0])
        time_index_1 = times.index(e[2])
        layer_index_1 = layers.index(e[4])
        row_index = betweenness_index(n_nodes, n_times, node_index_1, time_index_1, layer_index_1)
        node_index_2 = nodes.index(e[1])
        time_index_2 = times.index(e[3])
        layer_index_2 = layer_index_1
        column_index = betweenness_index(n_nodes, n_times, node_index_2, time_index_2, layer_index_2)
        weight_index = alpha + (1 - alpha) * (time_index_2 - time_index_1)
        static_edges.append((row_index, column_index, weight_index))
        static_data.append(weight_index)
        static_row.append(row_index)
        static_column.append(column_index)
        for layer in layers:
            if e[1] in node_layer_time_dict.keys():
                if layer in node_layer_time_dict[e[1]].keys():
                    temp = node_layer_time_dict[e[1]][layer]
                    t = [s for s in temp if s > e[3]]  # find, if exists, the time of next departure from that node
                    if t:

                        t = min(t)

                        # create switching links

                        row_switch_idx = column_index
                        node_switch_idx_2 = node_index_2
                        time_switch_idx_2 = time_index_2
                        layer_switch_idx_2 = layers.index(layer)
                        column_switch_idx = betweenness_index(n_nodes, n_times, node_switch_idx_2,
                                                              time_switch_idx_2, layer_switch_idx_2)
                        weight_switch_idx = alpha * epsilon
                        if layer_switch_idx_2 != layer_index_2:
                            static_edges.append((row_switch_idx, column_switch_idx, weight_switch_idx))
                            static_data.append(weight_switch_idx)
                            static_row.append(row_switch_idx)
                            static_column.append(column_switch_idx)
                        # create waiting links

                        time_waiting_idx_1 = time_switch_idx_2
                        row_waiting_idx = column_switch_idx
                        node_waiting_idx_2 = node_switch_idx_2
                        time_waiting_idx_2 = times.index(t)
                        layer_waiting_idx_2 = layer_switch_idx_2
                        column_waiting_idx = betweenness_index(n_nodes, n_times, node_waiting_idx_2,
                                                               time_waiting_idx_2, layer_waiting_idx_2)
                        weight_waiting_idx = time_waiting_idx_2 - time_waiting_idx_1
                        static_edges.append((row_waiting_idx, column_waiting_idx, weight_waiting_idx))
                        static_data.append(weight_waiting_idx)
                        static_row.append(row_waiting_idx)
                        static_column.append(column_waiting_idx)

    # create static network

    G = nx.DiGraph()
    G.add_weighted_edges_from(static_edges)
    # initialize betweenness as a dict with nodes as keys and zeros as values

    betweenness = zeros(n_nodes)

    # for every vertex v and starting time t, compute the best paths towards any other node u
    # every node has a triple: (node_index, time_index, layer_index) inverse_betweenness_index(entrance)
    for v in range(n_nodes):
        for t in range(n_times):
            v_t_family = betweenness_node_time_family(n_nodes, n_times, n_layers, v, t)
            #  delete missing sources
            v_t_family = list(set(v_t_family) & set(static_row))
            for i in v_t_family:
                paths = nx.single_source_dijkstra(G, i)
                for u in range(n_nodes):
                    if u != v:
                        u_family = betweenness_node_family(n_nodes, n_times, n_layers, u)
                        vu_distances = {}
                        for j in u_family:
                            if j in paths[0].keys():
                                if paths[0][j] in vu_distances.keys():
                                    vu_distances[paths[0][j]].append(paths[1][j])
                                else:
                                    vu_distances[paths[0][j]] = [paths[1][j]]
                        if vu_distances != {}:
                            min_distance = min(vu_distances.keys())
                            best_uv_paths = vu_distances[min_distance]
                            temp_betweenness = zeros(n_nodes)
                            if len(best_uv_paths) == 1:  # only one best path between u and v
                                if len(best_uv_paths[0]) > 2:
                                    start = inverse_betweenness_index(n_nodes, n_times,
                                                                      best_uv_paths[0][0])[0]
                                    arrive = inverse_betweenness_index(n_nodes, n_times,
                                                                       best_uv_paths[0][-1])[0]
                                    second_stop = inverse_betweenness_index(n_nodes, n_times,
                                                                            best_uv_paths[0][1])[0]
                                    if second_stop != start:
                                        for stop_index in range(1, len(best_uv_paths[0]) - 1):
                                            stop = inverse_betweenness_index(n_nodes, n_times,
                                                                             best_uv_paths[0][stop_index])[0]
                                            if stop != start and stop != arrive:
                                                temp_betweenness[stop] = 1
                            else:  # multiple best paths between u and v
                                count_false_best = 0  # count and delete paths which start with a waiting link
                                for best in best_uv_paths:
                                    temp_temp_betweenness = zeros(n_nodes)
                                    if len(best) > 2:
                                        start = inverse_betweenness_index(n_nodes, n_times,
                                                                          best[0])[0]
                                        arrive = inverse_betweenness_index(n_nodes, n_times,
                                                                           best[-1])[0]
                                        second_stop = inverse_betweenness_index(n_nodes, n_times,
                                                                                best[1])[0]
                                        if second_stop != start:
                                            for stop_index in range(1, len(best) - 1):

                                                stop = inverse_betweenness_index(n_nodes, n_times,
                                                                                 best[stop_index])[0]
                                                if stop != start and stop != arrive:
                                                    temp_temp_betweenness[stop] = 1
                                        else:
                                            count_false_best += 1
                                    temp_betweenness = temp_betweenness + temp_temp_betweenness
                                temp_betweenness = temp_betweenness / (len(best_uv_paths) - count_false_best)
                            betweenness = betweenness + temp_betweenness
    betweenness = dict(zip(nodes, betweenness))
    return betweenness
