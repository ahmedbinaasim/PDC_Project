#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential implementation of the algorithm for constructing n-1 independent spanning trees
in bubble-sort networks as described in the paper by Kao et al.

Command to Run the Sequential Implementation

python sequential.py --dimension 4 --visualize --verify --performance

This command will:

- Create a bubble-sort network B4 (dimension 4)
- Visualize the network structure
- Construct 3 independent spanning trees
- Verify their independence
- Visualize each of the trees
- Perform and display performance analysis
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import itertools
from tqdm import tqdm

class BubbleSortNetwork:
    """
    A class representing a bubble-sort network Bn.
    
    Attributes:
        n (int): The dimension of the bubble-sort network.
        vertices (list): All permutations of {1, 2, ..., n}.
    """
    
    def __init__(self, n):
        """
        Initialize a bubble-sort network of dimension n.
        
        Args:
            n (int): The dimension of the bubble-sort network.
        """
        self.n = n
        self.vertices = list(itertools.permutations(range(1, n+1)))
        self.vertex_map = {v: i for i, v in enumerate(self.vertices)}
        print(f"Created bubble-sort network B{n} with {len(self.vertices)} vertices.")
        
    def get_adjacent_vertices(self, v):
        """
        Get all adjacent vertices of a given vertex v.
        
        Args:
            v (tuple): A vertex (permutation) in the bubble-sort network.
            
        Returns:
            list: A list of adjacent vertices of v.
        """
        adjacent = []
        for i in range(self.n - 1):
            # Swap symbols at positions i and i+1
            new_v = list(v)
            new_v[i], new_v[i+1] = new_v[i+1], new_v[i]
            adjacent.append(tuple(new_v))
        return adjacent
    
    def create_graph(self):
        """
        Create a NetworkX graph representation of the bubble-sort network.
        
        Returns:
            nx.Graph: A NetworkX graph representing the bubble-sort network.
        """
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        
        for v in self.vertices:
            for u in self.get_adjacent_vertices(v):
                G.add_edge(v, u)
        
        return G
    
    def find_position(self, v, t):
        """
        Implementation of the FindPosition function from the paper.
        
        Args:
            v (tuple): A vertex in the bubble-sort network.
            t (int): The index of the tree.
            
        Returns:
            tuple: The parent of v in the t-th tree.
        """
        v_list = list(v)
        
        # Rule (1.1): if t = 2 and Swap(v, t) = 1n then p = Swap(v, t - 1)
        if t == 2:
            # Check if Swap(v, t) is the identity permutation
            swapped = list(v)
            pos_t = v_list.index(t)
            if pos_t < self.n - 1:  # Ensure valid swap
                swapped[pos_t], swapped[pos_t + 1] = swapped[pos_t + 1], swapped[pos_t]
                if tuple(swapped) == tuple(range(1, self.n+1)):
                    pos_t_minus_1 = v_list.index(t - 1)
                    if pos_t_minus_1 < self.n - 1:  # Ensure valid swap
                        v_list[pos_t_minus_1], v_list[pos_t_minus_1 + 1] = v_list[pos_t_minus_1 + 1], v_list[pos_t_minus_1]
                        return tuple(v_list)
        
        # Rule (1.2): if vn−1 ∈ {t, n−1} then j = r(v), p = Swap(v, j)
        if v[-2] in {t, self.n - 1}:
            # Find rightmost symbol j that is not in the right position
            j = None
            for i in range(self.n-1, -1, -1):
                if v[i] != i+1:
                    j = v[i]
                    break
            
            if j is not None:
                pos_j = v_list.index(j)
                if pos_j < self.n - 1:  # Ensure valid swap
                    v_list[pos_j], v_list[pos_j + 1] = v_list[pos_j + 1], v_list[pos_j]
                    return tuple(v_list)
        
        # Rule (1.3): p = Swap(v, t)
        pos_t = v_list.index(t)
        if pos_t < self.n - 1:  # Ensure valid swap
            v_list[pos_t], v_list[pos_t + 1] = v_list[pos_t + 1], v_list[pos_t]
            return tuple(v_list)
        
        # Fallback (if t is at the last position)
        return v
    
    def swap(self, v, x):
        """
        Implementation of the Swap function from the paper.
        
        Args:
            v (tuple): A vertex in the bubble-sort network.
            x (int): The symbol to swap with its successor.
            
        Returns:
            tuple: The vertex after swapping x with its successor.
        """
        v_list = list(v)
        pos_x = v_list.index(x)
        
        if pos_x < self.n - 1:  # Ensure valid swap
            v_list[pos_x], v_list[pos_x + 1] = v_list[pos_x + 1], v_list[pos_x]
            
        return tuple(v_list)
    
    def parent1(self, v, t):
        """
        Implementation of the Parent1 function from the paper.
        Determines the parent of vertex v in the t-th tree.
        
        Args:
            v (tuple): A vertex in the bubble-sort network.
            t (int): The index of the tree.
            
        Returns:
            tuple: The parent of v in the t-th tree.
        """
        if v[-1] == self.n:  # vn = n
            if t != self.n - 1:  # t ≠ n - 1
                return self.find_position(v, t)
            else:  # t = n - 1
                return self.swap(v, v[-2])
        else:
            if v[-1] == self.n - 1 and v[-2] == self.n and self.swap(v, self.n) != tuple(range(1, self.n+1)):
                if t == 1:
                    return self.swap(v, self.n)
                else:
                    return self.swap(v, t - 1)
            else:
                if v[-1] == t:
                    return self.swap(v, self.n)
                else:
                    return self.swap(v, t)
    
    def construct_ist(self, t, root=None):
        """
        Construct the t-th independent spanning tree rooted at the given root.
        
        Args:
            t (int): The index of the tree (1 <= t <= n-1).
            root (tuple, optional): The root of the tree. Defaults to the identity permutation.
            
        Returns:
            nx.DiGraph: A directed graph representing the t-th IST.
        """
        if root is None:
            root = tuple(range(1, self.n+1))  # Identity permutation
        
        tree = nx.DiGraph()
        tree.add_node(root)  # Add root
        
        # Initialize queue with vertices (except root)
        unprocessed = [v for v in self.vertices if v != root]
        
        # For each vertex, find its parent in the t-th tree
        parent_map = {}
        for v in unprocessed:
            parent = v
            while parent != root:
                parent = self.parent1(parent, t)
                if parent in parent_map:
                    parent = parent_map[parent]
                else:
                    parent_map[v] = parent
                    break
            
            tree.add_edge(parent, v)
            
        return tree
    
    def construct_all_ists(self, root=None):
        """
        Construct all n-1 independent spanning trees rooted at the given root.
        
        Args:
            root (tuple, optional): The root of the trees. Defaults to the identity permutation.
            
        Returns:
            list: A list of n-1 independent spanning trees.
        """
        print(f"Constructing {self.n-1} independent spanning trees...")
        start_time = time.time()
        
        trees = []
        for t in tqdm(range(1, self.n)):
            tree = self.construct_ist(t, root)
            trees.append(tree)
            
        elapsed_time = time.time() - start_time
        print(f"Construction completed in {elapsed_time:.4f} seconds.")
        
        return trees
    
    def verify_independence(self, trees, root=None):
        """
        Verify that the constructed spanning trees are independent.
        
        Args:
            trees (list): A list of spanning trees to verify.
            root (tuple, optional): The root of the trees. Defaults to the identity permutation.
            
        Returns:
            bool: True if the trees are independent, False otherwise.
        """
        if root is None:
            root = tuple(range(1, self.n+1))  # Identity permutation
        
        print("Verifying independence of spanning trees...")
        
        # Check each pair of trees
        for i in range(len(trees)):
            for j in range(i+1, len(trees)):
                tree1, tree2 = trees[i], trees[j]
                
                # Check each vertex
                for v in self.vertices:
                    if v == root:
                        continue
                    
                    # Get paths from v to root in both trees
                    path1 = nx.shortest_path(tree1, v, root)
                    path2 = nx.shortest_path(tree2, v, root)
                    
                    # Check for common vertices other than v and root
                    common_vertices = set(path1[1:-1]).intersection(set(path2[1:-1]))
                    if common_vertices:
                        print(f"Trees {i+1} and {j+1} share vertices in paths from {v} to root: {common_vertices}")
                        return False
                    
                    # Check for common edges
                    edges1 = set(zip(path1[:-1], path1[1:]))
                    edges2 = set(zip(path2[:-1], path2[1:]))
                    common_edges = edges1.intersection(edges2)
                    if common_edges:
                        print(f"Trees {i+1} and {j+1} share edges in paths from {v} to root: {common_edges}")
                        return False
        
        print("All trees are independent!")
        return True
    
    def visualize_network(self, highlight_vertices=None):
        """
        Visualize the bubble-sort network.
        
        Args:
            highlight_vertices (list, optional): A list of vertices to highlight. Defaults to None.
        """
        if self.n > 4:
            print("Network is too large to visualize. Skipping visualization.")
            return
        
        G = self.create_graph()
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(12, 10))
        
        # Draw all nodes and edges
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        if highlight_vertices:
            # Draw highlighted nodes
            highlight_nodes = set(highlight_vertices)
            nx.draw_networkx_nodes(G, pos, nodelist=list(highlight_nodes), node_color="red", node_size=500)
            
            # Draw other nodes
            other_nodes = set(G.nodes()) - highlight_nodes
            nx.draw_networkx_nodes(G, pos, nodelist=list(other_nodes), node_color="skyblue", node_size=300)
        else:
            nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=300)
        
        # Draw labels
        labels = {v: ''.join(map(str, v)) for v in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title(f"Bubble-Sort Network B{self.n}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"bubble_sort_network_B{self.n}.png", dpi=300)
        plt.show()
    
    def visualize_tree(self, tree, title=None):
        """
        Visualize a spanning tree of the bubble-sort network.
        
        Args:
            tree (nx.DiGraph): A directed graph representing a spanning tree.
            title (str, optional): The title of the plot. Defaults to None.
        """
        if self.n > 4:
            print("Tree is too large to visualize. Skipping visualization.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Use hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(tree, prog='dot') if nx.nx_agraph.graphviz_layout else nx.spring_layout(tree)
        
        # Draw nodes and edges
        nx.draw_networkx_edges(tree, pos, arrows=True)
        nx.draw_networkx_nodes(tree, pos, node_color="skyblue", node_size=300)
        
        # Highlight root
        root = tuple(range(1, self.n+1))  # Identity permutation
        nx.draw_networkx_nodes(tree, pos, nodelist=[root], node_color="red", node_size=500)
        
        # Draw labels
        labels = {v: ''.join(map(str, v)) for v in tree.nodes()}
        nx.draw_networkx_labels(tree, pos, labels=labels, font_size=8)
        
        if title:
            plt.title(title)
        else:
            plt.title("Spanning Tree of Bubble-Sort Network")
        
        plt.axis('off')
        plt.tight_layout()
        
        if title:
            plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
        
        plt.show()
    
    def visualize_all_trees(self, trees):
        """
        Visualize all spanning trees.
        
        Args:
            trees (list): A list of spanning trees to visualize.
        """
        for i, tree in enumerate(trees):
            self.visualize_tree(tree, f"Independent Spanning Tree T_{i+1}")
    
    def performance_analysis(self):
        """
        Perform performance analysis of the sequential algorithm.
        """
        print("\nPerformance Analysis:")
        print("-" * 50)
        
        # Measure time to construct all trees
        start_time = time.time()
        trees = self.construct_all_ists()
        construction_time = time.time() - start_time
        
        # Measure time to verify independence
        start_time = time.time()
        self.verify_independence(trees)
        verification_time = time.time() - start_time
        
        print(f"Network Dimension: B{self.n}")
        print(f"Number of Vertices: {len(self.vertices)}")
        print(f"Number of Trees: {self.n-1}")
        print(f"Construction Time: {construction_time:.4f} seconds")
        print(f"Verification Time: {verification_time:.4f} seconds")
        print(f"Total Time: {construction_time + verification_time:.4f} seconds")
        
        # Additional performance metrics
        avg_time_per_tree = construction_time / (self.n-1)
        avg_time_per_vertex_per_tree = construction_time / (len(self.vertices) * (self.n-1))
        
        print(f"Average Time per Tree: {avg_time_per_tree:.6f} seconds")
        print(f"Average Time per Vertex per Tree: {avg_time_per_vertex_per_tree:.8f} seconds")
        
        # Visualize performance metrics
        self._plot_performance_metrics(construction_time, verification_time)
        
        return {
            "dimension": self.n,
            "vertices": len(self.vertices),
            "trees": self.n-1,
            "construction_time": construction_time,
            "verification_time": verification_time,
            "total_time": construction_time + verification_time,
            "avg_time_per_tree": avg_time_per_tree,
            "avg_time_per_vertex_per_tree": avg_time_per_vertex_per_tree
        }
    
    def _plot_performance_metrics(self, construction_time, verification_time):
        """
        Plot performance metrics.
        
        Args:
            construction_time (float): Time taken to construct all trees.
            verification_time (float): Time taken to verify independence.
        """
        # Bar chart of time distribution
        plt.figure(figsize=(10, 6))
        
        labels = ['Construction', 'Verification', 'Total']
        times = [construction_time, verification_time, construction_time + verification_time]
        
        plt.bar(labels, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        for i, v in enumerate(times):
            plt.text(i, v + 0.1, f"{v:.4f}s", ha='center')
        
        plt.ylabel('Time (seconds)')
        plt.title(f'Performance Metrics for B{self.n}')
        plt.tight_layout()
        plt.savefig(f"performance_metrics_B{self.n}.png", dpi=300)
        plt.show()


def main():
    """
    Main function to demonstrate the sequential algorithm.
    """
    print("Bubble-Sort Network IST Construction Algorithm (Sequential Version)")
    print("=" * 70)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Construct independent spanning trees in bubble-sort networks.')
    parser.add_argument('--dimension', '-d', type=int, default=4, help='Dimension of the bubble-sort network (default: 4)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize the network and trees')
    parser.add_argument('--performance', '-p', action='store_true', help='Perform performance analysis')
    parser.add_argument('--verify', action='store_true', help='Verify independence of trees')
    
    args = parser.parse_args()
    
    n = args.dimension
    if n < 3:
        print("Error: Dimension must be at least 3.")
        return
    
    # Create bubble-sort network
    bn = BubbleSortNetwork(n)
    
    # Visualize network if requested
    if args.visualize:
        bn.visualize_network()
    
    # Construct independent spanning trees
    trees = bn.construct_all_ists()
    
    # Verify independence if requested
    if args.verify:
        bn.verify_independence(trees)
    
    # Visualize trees if requested
    if args.visualize:
        bn.visualize_all_trees(trees)
    
    # Perform performance analysis if requested
    if args.performance:
        bn.performance_analysis()


if __name__ == "__main__":
    main()