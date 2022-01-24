import numpy as np
#import heapq
from typing import Union

class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """ Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """ Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. 
        Note that because we assume our input graph is undirected, `self.adj_mat` is symmetric. 
        Row i and column j represents the edge weight between vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        TODO: 
            This function does not return anything. Instead, store the adjacency matrix 
        representation of the minimum spanning tree of `self.adj_mat` in `self.mst`.
        We highly encourage the use of priority queues in your implementation. See the heapq
        module, particularly the `heapify`, `heappop`, and `heappush` functions.
        """
        
        adjacency_mat = self.adj_mat
        no_nodes = len(adjacency_mat) #compute the total number of nodes
        mst = np.zeros((no_nodes,no_nodes)) #generate an empty matrix to become mst
        nodes = list([int(x) for x in np.linspace(0,no_nodes-1,no_nodes)]) #list all nodes
        V = set(nodes) #create a set of all vertices
        section = set([nodes[0]]) #initialize section 
        
        while section != V:
            edge_weights = [] #generate empty vectors to hold edge weights and edges for all edges connecting section with V-section 
            edges = []
            for node1 in section: #loop through all nodes in section and check if they connect with V-section
                for node2 in set([node for node in V if node not in section]):
                    if adjacency_mat[node1,node2] != 0:
                        edge_weights.append(adjacency_mat[node1,node2]) #if the nodes are connected, append edge and edge_weight
                        edges.append((node1,node2))
        
            local_min = min(edge_weights) #find the shortest edge between section and V-section and the nodes forming this edge
            v1,v2 = edges[edge_weights.index(local_min)] 
            mst[v1,v2] = local_min #update mst
            mst[v2,v1] = local_min
            section.add(v2) #update section set
        
        
        
        
        self.mst = mst
        

