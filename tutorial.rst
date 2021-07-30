********
Tutorial
********

_________

NetworkSNS is built upon NetworkX_ and DyNetX_ and is designed to configure, model and analyze complex and dynamic networks.

In this tutorial we will introduce some of the basic centrality measures:

Total communicability
---------------------


We compute the total communicability of a simple graph, that is the row sum of the exponential of the adjacency matrix :math:`A` .

.. code:: python

   from networksns import centrality_measures as cm
   import networkx as nx

    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    tc = cm.total_communicability(G)

Adding the parameter ``t`` we can also compute the exponential matrix of :math:`tA`.

.. code:: python

   t = 3
   tc = cm.total_communicability(G)


.. _NetworkX: https://networkx.github.io
.. _DyNetX: https://dynetx.readthedocs.io