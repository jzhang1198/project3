# write tests for bfs
import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """ Helper function to check the correctness of the adjacency matrix encoding an MST.
        Note that because the MST of a graph is not guaranteed to be unique, we cannot 
        simply check for equality against a known MST of a graph. 

        Arguments:
            adj_mat: Adjacency matrix of full graph
            mst: Adjacency matrix of proposed minimum spanning tree
            expected_weight: weight of the minimum spanning tree of the full graph
            allowed_error: Allowed difference between proposed MST weight and `expected_weight`

        TODO: 
            Add additional assertions to ensure the correctness of your MST implementation
        For example, how many edges should a minimum spanning tree have? Are minimum spanning trees
        always connected? What else can you think of?
    """
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    
    no_nodes = len(adj_mat)
    no_edges = np.count_nonzero(mst) / 2 #count the number of nonzero elements in mst. since this quantity represents twice the number of edges, dividing it by two will yield the number of edges in the mst
    assert no_edges == no_nodes - 1 #a mst will contain n-1 edges, where n is the number of nodes.
    
    rows,columns = np.nonzero(mst) #convert the adjacency matrix of the mst to that of an unweighted graph with the same connectivity 
    unweighted = mst
    for ind in range(0,len(rows)):
        r = rows[ind]
        c = columns[ind]
        unweighted[r,c] = 1
    D = np.zeros((no_nodes,no_nodes)) #compute degree matrix corresponding to the MST
    for node in range(0,no_nodes):
        D[node,node] = np.count_nonzero(unweighted[:,node]) 
        
    laplacian = D - unweighted #compute the laplacian matrix corresponding to the unweighted MST
    eigs = np.sort(np.linalg.eig(laplacian)[0])[::-1]
    fiedler = eigs[-2] #if the MST is connected, the second smallest eigenvalue of its laplacian will be nonzero. https://en.wikipedia.org/wiki/Algebraic_connectivity
    assert fiedler > 0
    

def test_mst_small():
    """ Unit test for the construction of a minimum spanning tree on a small graph """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """ Unit test for the construction of a minimum spanning tree using 
    single cell data, taken from the Slingshot R package 
    (https://bioconductor.org/packages/release/bioc/html/slingshot.html)
    """
    file_path = './data/slingshot_example.txt'
    # load coordinates of single cells in low-dimensional subspace
    coords = np.loadtxt(file_path)
    # compute pairwise distances for all 140 cells to form an undirected weighted graph
    dist_mat = pairwise_distances(coords)
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """ TODO: Write at least one unit test for MST construction """
    
    #dummy network to test this 
    file_path = './data/dummy.csv' #path to a connected graph of three nodes
    g = Graph(file_path)
    g.construct_mst()
    
    shape = np.shape(g.mst)
    
    assert shape == (3,3) #ensure that the mst adjacency matrix is square
    
    for row_ind in range(0,shape[0]):
        for column_ind in range(0,shape[1]):
            assert g.mst[row_ind,column_ind] == g.mst[column_ind,row_ind] #check that the mst adjacency matrix is symmetric
    
    
