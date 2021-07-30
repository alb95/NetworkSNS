import unittest
import dynetx as dn
import networkx as nx
from networksns import centrality_measures as cm
import numpy as np


class NetworkSNSTestCase(unittest.TestCase):

    def test_graph_slice(self):
        g = dn.DynGraph()
        g.add_interaction(1, 2, 2)
        g.add_interaction(1, 2, 2, e=6)
        g.add_interaction(1, 2, 7, e=11)
        g.add_interaction(1, 2, 8, e=15)
        h = cm.graph_slice(g, 3)
        self.assertIsInstance(h, nx.Graph)
        self.assertEqual(h.number_of_nodes(), 2)
        self.assertEqual(h.number_of_edges(), 1)

    def test_broadcast_centrality(self):
        g = dn.DynGraph()
        g.add_interaction(1, 2, 2, 5)
        g.add_interaction(1, 3, 2, 5)
        g.add_interaction(2, 3, 4)
        bc, alpha = cm.broadcast_centrality(g)
        self.assertAlmostEqual(bc[1], 0.701091807280032, 10)
        self.assertAlmostEqual(bc[2], 0.504217352817621, 10)
        self.assertEqual(alpha, 0.45)
        bc, alpha = cm.broadcast_centrality(g, alpha=0.45)
        self.assertAlmostEqual(bc[1], 0.701091807280032, 10)
        self.assertAlmostEqual(bc[2], 0.504217352817621, 10)

    def test_receive_centrality(self):
        g = dn.DynGraph()
        g.add_interaction(1, 2, 2, 5)
        g.add_interaction(1, 'node', 2, 5)
        g.add_interaction(2, 'node', 4)
        rc, alpha = cm.receive_centrality(g)
        self.assertAlmostEqual(rc[1], 0.586484236968094, 10)
        self.assertAlmostEqual(rc['node'], 0.572728661666217, 10)
        self.assertEqual(alpha, 0.45)
        rc, alpha = cm.receive_centrality(g, alpha=0.45)
        self.assertAlmostEqual(rc[1], 0.586484236968094, 10)
        self.assertAlmostEqual(rc['node'], 0.572728661666217, 10)

    def test_approximated_broadcast_centrality(self):
        g = dn.DynGraph()
        g.add_interaction(1, 2, 2, 5)
        g.add_interaction(1, 'node', 2, 5)
        g.add_interaction(2, 'node', 4)
        bc, alpha = cm.approximated_broadcast_centrality(g)
        self.assertAlmostEqual(bc[1], 0.701091807280032, 10)
        self.assertAlmostEqual(bc['node'], 0.504217352817621, 10)
        self.assertEqual(alpha, 0.45)
        bc = cm.approximated_broadcast_centrality(g, alpha=0.45)
        self.assertAlmostEqual(bc[1], 0.701091807280032, 10)
        self.assertAlmostEqual(bc['node'], 0.504217352817621, 10)
        self.assertEqual(alpha, 0.45)

    def test_approximated_receive_centrality(self):
        g = dn.DynGraph()
        g.add_interaction(1, 2, 2, 5)
        g.add_interaction(1, 'node', 2, 5)
        g.add_interaction(2, 'node', 4)
        rc, alpha = cm.approximated_receive_centrality(g)
        self.assertAlmostEqual(rc[1], 0.615160904649326, 10)
        self.assertAlmostEqual(rc['node'], 0.557484108020589, 10)
        self.assertEqual(alpha, 0.45)
        rc = cm.approximated_receive_centrality(g, alpha=0.45)
        self.assertAlmostEqual(rc[1], 0.615160904649326, 10)
        self.assertAlmostEqual(rc['node'], 0.557484108020589, 10)
        self.assertEqual(alpha, 0.45)

    def test_exponential_symmetric_quadrature(self):
        A = np.arange(0, 1, 0.01)
        A = A.reshape(10, 10)
        A = A + A.transpose()
        u = np.arange(0, 10)
        q = cm.exponential_symmetric_quadrature(A, u)
        self.assertAlmostEqual(q, 13165244.30434994, 2)
        q = cm.exponential_symmetric_quadrature(A, u, tol=1e-8)
        self.assertAlmostEqual(q, 13165244.30434994, 2)
        q = cm.exponential_symmetric_quadrature(A, u, tol=1e-8, maxit=3)
        self.assertAlmostEqual(q, 13165244.30434994, 2)

    def test_exponential_quadrature(self):
        A = np.arange(0, 1, 0.01)
        A = A.reshape(10, 10)
        A = A + A.transpose()
        u = np.arange(0, 10)
        v = np.ones(10)
        q = cm.exponential_quadrature(A, u, v)
        self.assertAlmostEqual(q, 2466072.870541437, 2)
        q = cm.exponential_quadrature(A, u, v, tol=1e-8)
        self.assertAlmostEqual(q, 2466072.870541437, 2)
        q = cm.exponential_quadrature(A, u, v, tol=1e-8, maxit=3)
        self.assertAlmostEqual(q, 2466072.870541437, 2)

    def test_total_communicability(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        tc = cm.total_communicability(g)
        self.assertAlmostEqual(tc[1], 3.546482428617160, 8)
        tc = cm.total_communicability(g, t=3)
        self.assertAlmostEqual(tc[1], 59.402061428302126, 8)

    def test_node_total_communicability(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        tc_node_1 = cm.node_total_communicability(g, 1)
        self.assertAlmostEqual(tc_node_1, 3.546482428617160, 8)
        tc_node_2 = cm.node_total_communicability(g, 2, t=3)
        self.assertAlmostEqual(tc_node_2, 84.001248823238072, 8)
        tc_node_3 = cm.node_total_communicability(g, 3, t=3, tol=1e-5)
        self.assertAlmostEqual(tc_node_3, 59.402061428302119, 8)
        tc_node_3 = cm.node_total_communicability(g, 3, t=3, tol=1e-5, maxit=3)
        self.assertAlmostEqual(tc_node_3, 59.402061428302119, 8)

    def test_total_network_communicability(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        tc = cm.total_network_communicability(g)
        self.assertAlmostEqual(tc, 12.007746157860071, 8)
        tc = cm.total_network_communicability(g, t=3)
        self.assertAlmostEqual(tc, 202.8053716798423, 7)
        tc = cm.total_network_communicability(g, t=3, tol=1e-5)
        self.assertAlmostEqual(tc, 202.8053716798423, 7)
        tc = cm.total_network_communicability(g, t=3, tol=1e-5, maxit=6)
        self.assertAlmostEqual(tc, 202.8053716798423, 7)

    def test_edge_total_communicability(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        tc_edge_12 = cm.edge_total_communicability(g, 1, 2)
        self.assertAlmostEqual(tc_edge_12, 17.430185523165417, 8)
        tc_edge_12 = cm.edge_total_communicability(g, 1, 2, t=3)
        self.assertAlmostEqual(tc_edge_12, 4989.847342652080, 7)
        tc_edge_12 = cm.edge_total_communicability(g, 1, 2, t=3, tol=1e-5)
        self.assertAlmostEqual(tc_edge_12, 4989.847342652080, 7)
        tc_edge_12 = cm.edge_total_communicability(g, 1, 2, t=3, tol=1e-5, maxit=6)
        self.assertAlmostEqual(tc_edge_12, 4989.847342652080, 7)

    def test_total_directed_communicability(self):
        g = nx.DiGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        thc, tac = cm.total_directed_communicability(g)
        self.assertAlmostEqual(thc, 4.086161269630487, 6)
        self.assertAlmostEqual(tac, 4.086161269630487, 6)
        thc, tac = cm.total_directed_communicability(g, t=3)
        self.assertAlmostEqual(thc, 21.13532399155552, 6)
        self.assertAlmostEqual(tac, 21.13532399155552, 6)

    def test_node_total_directed_communicability(self):
        g = nx.DiGraph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        thc, tac = cm.node_total_directed_communicability(g, 1)
        self.assertAlmostEqual(thc, 2.9486638948883104, 8)
        thc, tac = cm.node_total_directed_communicability(g, 1, t=3)
        self.assertAlmostEqual(thc, 75.61539136707464, 8)
        thc, tac = cm.node_total_directed_communicability(g, 1, t=3)
        self.assertAlmostEqual(thc, 75.61539136707464, 8)
        thc, tac = cm.node_total_directed_communicability(g, 1, t=3, tol=1e-5)
        self.assertAlmostEqual(thc, 75.61539136707464, 8)
        thc, tac = cm.node_total_directed_communicability(g, 1, t=3, tol=1e-5, maxit=6)
        self.assertAlmostEqual(tac, 10.0178749274099034, 8)

    def test_trip_centrality(self):

        multilayer_edge_list = [['u', 'v', 1, 3, 'l'], ['u', 'w', 1, 2, 'l'], ['w', 'u', 1, 5, 'm'],
                                ['u', 'v', 2, 4, 'm'], ['u', 'w', 3, 4, 'm'], ['w', 'u', 4, 5, 'r'],
                                ['u', 'v', 1, 5, 'm'], ['v', 'w', 2, 3, 'm'], ['v', 'u', 4, 5, 'r']]
        tc = cm.trip_centrality(multilayer_edge_list, 0.3, 0.2)
        self.assertAlmostEqual(tc['v'], 3.7983084684606423, 8)

    def test_betweenness_centrality(self):

        edge_list = [[1, 2, 1, 2, 'lambda'], [3, 2, 1, 8, 'lambda'], [1, 2, 3, 5, 'mu'], [3, 1, 1, 2, 'mu'],
                     [2, 4, 6, 7, 'rho']]
        bc = cm.betweenness_centrality(edge_list, 1, 0.5, 0.4)
        self.assertAlmostEqual(bc[3], 0)
        self.assertAlmostEqual(bc[1], 2)


if __name__ == '__main__':
    unittest.main()
