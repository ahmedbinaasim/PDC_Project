#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel implementation of the algorithm for constructing n-1 independent spanning trees
in bubble-sort networks as described in the paper by Kao et al.

This implementation uses MPI for inter-node parallelism and multiprocessing/threading
for intra-node parallelism (simulating OpenMP functionality in Python).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import itertools
from tqdm import tqdm
import multiprocessing as mp
from mpi4py import MPI
import os
import sys
import json

# Import sequential implementation classes and functions for reuse
from sequential import BubbleSortNetwork

class ParallelBubbleSortNetwork(BubbleSortNetwork):
    """
    Extension of BubbleSortNetwork that implements parallel construction of ISTs
    using MPI for inter-node parallelism and multiprocessing for intra-node parallelism.
    """
    
    def __init__(self, n, comm=None):
        """
        Initialize a parallel bubble-sort network of dimension n.
        
        Args:
            n (int): The dimension of the bubble-sort network.
            comm (MPI.Comm, optional): MPI communicator. Defaults to MPI.COMM_WORLD.
        """
        super().__init__(n)
        
        # Set up MPI environment
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Get node information for hybrid parallelism
        self.hostname = MPI.Get_processor_name()
        self.num_cores = mp.cpu_count()
        
        if self.rank == 0:
            print(f"MPI process {self.rank}: Using {self.size} MPI processes")
            print(f"MPI process {self.rank}: Running on {self.hostname} with {self.num_cores} CPU cores")
        
        # Partition vertices among MPI processes
        self.local_vertices = self._distribute_vertices()
        
        if self.rank == 0:
            print(f"Vertices distributed across {self.size} MPI processes.")
            print(f"Average vertices per process: {len(self.vertices) / self.size:.1f}")
    
    def _distribute_vertices(self):
        """
        Distribute vertices among MPI processes using a simple block partitioning.
        
        Returns:
            list: List of vertices assigned to this MPI process.
        """
        vertices_list = list(self.vertices)
        n_vertices = len(vertices_list)
        
        # Simple block partitioning
        chunk_size = n_vertices // self.size
        start = self.rank * chunk_size
        end = start + chunk_size if self.rank < self.size - 1 else n_vertices
        
        local_vertices = vertices_list[start:end]
        
        if self.rank == 0:
            print(f"MPI process {self.rank}: Vertices distributed. Process {self.rank} has {len(local_vertices)} vertices.")
        
        return local_vertices
    
    def _compute_parents_for_tree(self, t, vertices):
        """
        Compute parent relationships for a specific tree t and a subset of vertices.
        
        Args:
            t (int): The tree index.
            vertices (list): List of vertices to process.
            
        Returns:
            dict: Dictionary mapping vertices to their parents in tree t.
        """
        parent_map = {}
        root = tuple(range(1, self.n+1))  # Identity permutation
        
        for v in vertices:
            if v != root:
                parent = self.parent1(v, t)
                parent_map[v] = parent
        
        return parent_map
    
    def _process_trees_with_openmp(self, tree_indices, vertices):
        """
        Process multiple trees in parallel using multiprocessing (simulating OpenMP).
        
        Args:
            tree_indices (list): List of tree indices to process.
            vertices (list): List of vertices to process.
            
        Returns:
            dict: Dictionary mapping tree indices to parent maps.
        """
        results = {}
        
        # Determine number of processes based on available cores and trees
        num_processes = min(len(tree_indices), self.num_cores)
        
        if num_processes > 1:
            # Use multiprocessing for tree-level parallelism
            with mp.Pool(processes=num_processes) as pool:
                tasks = [(t, vertices) for t in tree_indices]
                pool_results = pool.starmap(self._compute_parents_for_tree, tasks)
                
                # Combine results
                for t, parent_map in zip(tree_indices, pool_results):
                    results[t] = parent_map
        else:
            # Sequential processing if only one tree or core
            for t in tree_indices:
                results[t] = self._compute_parents_for_tree(t, vertices)
        
        return results
    
    def construct_ist_parallel(self, t):
        """
        Construct the t-th independent spanning tree in parallel.
        
        Args:
            t (int): The index of the tree (1 <= t <= n-1).
            
        Returns:
            nx.DiGraph: A directed graph representing the t-th IST.
        """
        root = tuple(range(1, self.n+1))  # Identity permutation
        
        # Each process computes parent relationships for its local vertices
        local_parent_map = self._compute_parents_for_tree(t, self.local_vertices)
        
        # Gather parent maps from all processes
        all_parent_maps = self.comm.gather(local_parent_map, root=0)
        
        # Process 0 constructs the complete tree
        if self.rank == 0:
            # Combine all parent maps
            combined_parent_map = {}
            for parent_map in all_parent_maps:
                combined_parent_map.update(parent_map)
            
            # Construct the tree
            tree = nx.DiGraph()
            tree.add_node(root)  # Add root
            
            for v, parent in combined_parent_map.items():
                tree.add_edge(parent, v)
            
            return tree
        
        return None
    
    def construct_all_ists_parallel(self):
        """
        Construct all n-1 independent spanning trees in parallel.
        
        Returns:
            list: A list of n-1 independent spanning trees.
        """
        if self.rank == 0:
            print(f"Constructing {self.n-1} independent spanning trees in parallel...")
            overall_start_time = time.time()
        
        trees = []
        
        # Decide on parallelization strategy based on number of trees and processes
        if self.size >= self.n - 1:
            # If we have more processes than trees, each process handles one tree
            local_trees = []
            
            if self.rank < self.n - 1:
                t = self.rank + 1
                if self.rank == 0:
                    print(f"MPI process {self.rank}: Each process constructing one tree.")
                    
                local_tree = self.construct_ist_parallel(t)
                local_trees = [local_tree]
            
            # Gather trees from all processes
            all_trees = self.comm.gather(local_trees, root=0)
            
            if self.rank == 0:
                # Flatten the list of trees
                trees = [tree for sublist in all_trees for tree in sublist if tree is not None]
        else:
            # If we have more trees than processes, distribute trees among processes
            tree_indices = list(range(1, self.n))
            trees_per_process = len(tree_indices) // self.size
            
            start_idx = self.rank * trees_per_process
            end_idx = start_idx + trees_per_process if self.rank < self.size - 1 else len(tree_indices)
            
            local_tree_indices = tree_indices[start_idx:end_idx]
            
            if self.rank == 0:
                print(f"MPI process {self.rank}: Distributing {self.n-1} trees among {self.size} processes.")
                print(f"MPI process {self.rank}: Process {self.rank} handling trees {local_tree_indices}")
            
            # Create a progress bar only for the root process
            if self.rank == 0:
                local_tree_indices_tqdm = tqdm(local_tree_indices)
            else:
                local_tree_indices_tqdm = local_tree_indices
            
            # Each process constructs its assigned trees
            local_trees = []
            for t in local_tree_indices_tqdm:
                local_tree = self.construct_ist_parallel(t)
                if local_tree is not None:
                    local_trees.append((t, local_tree))
            
            # Gather trees from all processes
            all_trees = self.comm.gather(local_trees, root=0)
            
            if self.rank == 0:
                # Combine and sort trees by index
                all_trees_flat = sorted([item for sublist in all_trees for item in sublist], key=lambda x: x[0])
                trees = [tree for _, tree in all_trees_flat]
        
        if self.rank == 0:
            elapsed_time = time.time() - overall_start_time
            print(f"Parallel construction completed in {elapsed_time:.4f} seconds.")
        
        return trees
    
    def construct_all_ists_hybrid(self):
        """
        Construct all n-1 independent spanning trees using hybrid parallelism (MPI + OpenMP-like).
        
        Returns:
            list: A list of n-1 independent spanning trees.
        """
        if self.rank == 0:
            print(f"Constructing {self.n-1} independent spanning trees using hybrid parallelism...")
            overall_start_time = time.time()
        
        # Distribute tree indices among MPI processes
        tree_indices = list(range(1, self.n))
        trees_per_process = len(tree_indices) // self.size
        
        start_idx = self.rank * trees_per_process
        end_idx = start_idx + trees_per_process if self.rank < self.size - 1 else len(tree_indices)
        
        local_tree_indices = tree_indices[start_idx:end_idx]
        
        if self.rank == 0:
            print(f"MPI process {self.rank}: Distributing {self.n-1} trees among {self.size} processes.")
            print(f"MPI process {self.rank}: Each process using OpenMP-like parallelism with {min(self.num_cores, len(local_tree_indices))} threads.")
        
        # Use intra-node parallelism via multiprocessing to handle multiple trees
        local_parent_maps = self._process_trees_with_openmp(local_tree_indices, self.local_vertices)
        
        # Gather all parent maps from all processes
        all_parent_maps = self.comm.gather(local_parent_maps, root=0)
        
        trees = []
        if self.rank == 0:
            # Combine all parent maps and construct complete trees
            root = tuple(range(1, self.n+1))  # Identity permutation
            
            # Flatten and organize parent maps by tree index
            tree_parent_maps = {}
            for process_maps in all_parent_maps:
                for t, parent_map in process_maps.items():
                    if t not in tree_parent_maps:
                        tree_parent_maps[t] = {}
                    tree_parent_maps[t].update(parent_map)
            
            # Construct each tree
            for t in range(1, self.n):
                if t in tree_parent_maps:
                    tree = nx.DiGraph()
                    tree.add_node(root)  # Add root
                    
                    for v, parent in tree_parent_maps[t].items():
                        tree.add_edge(parent, v)
                    
                    trees.append(tree)
            
            elapsed_time = time.time() - overall_start_time
            print(f"Hybrid parallel construction completed in {elapsed_time:.4f} seconds.")
        
        return trees
    
    def performance_analysis_parallel(self):
        """
        Perform performance analysis of the parallel algorithm.
        
        Returns:
            dict: Dictionary containing performance metrics.
        """
        if self.rank == 0:
            print("\nPerformance Analysis (Parallel):")
            print("-" * 50)
        
        # Measure time to construct all trees
        self.comm.Barrier()  # Synchronize all processes
        start_time = time.time()
        
        trees_parallel = self.construct_all_ists_parallel()
        
        self.comm.Barrier()  # Synchronize all processes
        construction_time_parallel = time.time() - start_time
        
        # Measure time to construct all trees with hybrid parallelism
        self.comm.Barrier()  # Synchronize all processes
        start_time = time.time()
        
        trees_hybrid = self.construct_all_ists_hybrid()
        
        self.comm.Barrier()  # Synchronize all processes
        construction_time_hybrid = time.time() - start_time
        
        # Measure time to verify independence (only in root process)
        verification_time = 0
        if self.rank == 0 and trees_parallel:
            start_time = time.time()
            self.verify_independence(trees_parallel)
            verification_time = time.time() - start_time
        
        if self.rank == 0:
            print(f"Network Dimension: B{self.n}")
            print(f"Number of Vertices: {len(self.vertices)}")
            print(f"Number of Trees: {self.n-1}")
            print(f"MPI Processes: {self.size}")
            print(f"Threads per Process: {min(self.num_cores, self.n-1)}")
            print(f"Parallel Construction Time: {construction_time_parallel:.4f} seconds")
            print(f"Hybrid Construction Time: {construction_time_hybrid:.4f} seconds")
            print(f"Verification Time: {verification_time:.4f} seconds")
            
            # Calculate additional metrics
            avg_time_per_tree_parallel = construction_time_parallel / (self.n-1)
            avg_time_per_vertex_per_tree_parallel = construction_time_parallel / (len(self.vertices) * (self.n-1))
            
            avg_time_per_tree_hybrid = construction_time_hybrid / (self.n-1)
            avg_time_per_vertex_per_tree_hybrid = construction_time_hybrid / (len(self.vertices) * (self.n-1))
            
            print(f"Average Time per Tree (Parallel): {avg_time_per_tree_parallel:.6f} seconds")
            print(f"Average Time per Vertex per Tree (Parallel): {avg_time_per_vertex_per_tree_parallel:.8f} seconds")
            print(f"Average Time per Tree (Hybrid): {avg_time_per_tree_hybrid:.6f} seconds")
            print(f"Average Time per Vertex per Tree (Hybrid): {avg_time_per_vertex_per_tree_hybrid:.8f} seconds")
            
            # Visualize performance metrics
            self._plot_parallel_performance_metrics(construction_time_parallel, construction_time_hybrid, verification_time)
            
            return {
                "dimension": self.n,
                "vertices": len(self.vertices),
                "trees": self.n-1,
                "mpi_processes": self.size,
                "threads_per_process": min(self.num_cores, self.n-1),
                "construction_time_parallel": construction_time_parallel,
                "construction_time_hybrid": construction_time_hybrid,
                "verification_time": verification_time,
                "avg_time_per_tree_parallel": avg_time_per_tree_parallel,
                "avg_time_per_vertex_per_tree_parallel": avg_time_per_vertex_per_tree_parallel,
                "avg_time_per_tree_hybrid": avg_time_per_tree_hybrid,
                "avg_time_per_vertex_per_tree_hybrid": avg_time_per_vertex_per_tree_hybrid
            }
        
        return None
    
    def _plot_parallel_performance_metrics(self, construction_time_parallel, construction_time_hybrid, verification_time):
        """
        Plot performance metrics for parallel implementation.
        
        Args:
            construction_time_parallel (float): Time taken to construct all trees with MPI only.
            construction_time_hybrid (float): Time taken to construct all trees with MPI+OpenMP.
            verification_time (float): Time taken to verify independence.
        """
        # Bar chart of time distribution
        plt.figure(figsize=(10, 6))
        
        labels = ['MPI Only', 'MPI+OpenMP', 'Verification']
        times = [construction_time_parallel, construction_time_hybrid, verification_time]
        
        plt.bar(labels, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        for i, v in enumerate(times):
            plt.text(i, v + 0.1, f"{v:.4f}s", ha='center')
        
        plt.ylabel('Time (seconds)')
        plt.title(f'Parallel Performance Metrics for B{self.n} with {self.size} processes')
        plt.tight_layout()
        plt.savefig(f"parallel_performance_metrics_B{self.n}_P{self.size}.png", dpi=300)
        plt.show()
    
    def compare_with_sequential(self, sequential_metrics):
        """
        Compare parallel performance with sequential performance.
        
        Args:
            sequential_metrics (dict): Performance metrics from sequential implementation.
        """
        if self.rank != 0 or sequential_metrics is None:
            return
        
        parallel_metrics = self.performance_analysis_parallel()
        
        if parallel_metrics:
            print("\nPerformance Comparison (Sequential vs. Parallel):")
            print("-" * 60)
            
            # Calculate speedup
            speedup_parallel = sequential_metrics["construction_time"] / parallel_metrics["construction_time_parallel"]
            speedup_hybrid = sequential_metrics["construction_time"] / parallel_metrics["construction_time_hybrid"]
            
            # Calculate efficiency
            efficiency_parallel = speedup_parallel / self.size
            efficiency_hybrid = speedup_hybrid / self.size
            
            print(f"Sequential Construction Time: {sequential_metrics['construction_time']:.4f} seconds")
            print(f"Parallel Construction Time (MPI): {parallel_metrics['construction_time_parallel']:.4f} seconds")
            print(f"Parallel Construction Time (MPI+OpenMP): {parallel_metrics['construction_time_hybrid']:.4f} seconds")
            print(f"Speedup (MPI): {speedup_parallel:.2f}x")
            print(f"Speedup (MPI+OpenMP): {speedup_hybrid:.2f}x")
            print(f"Efficiency (MPI): {efficiency_parallel:.2f}")
            print(f"Efficiency (MPI+OpenMP): {efficiency_hybrid:.2f}")
            
            # Plot comparison
            self._plot_performance_comparison(
                sequential_metrics["construction_time"],
                parallel_metrics["construction_time_parallel"],
                parallel_metrics["construction_time_hybrid"],
                speedup_parallel,
                speedup_hybrid
            )
            
            return {
                "sequential_time": sequential_metrics["construction_time"],
                "parallel_time_mpi": parallel_metrics["construction_time_parallel"],
                "parallel_time_hybrid": parallel_metrics["construction_time_hybrid"],
                "speedup_mpi": speedup_parallel,
                "speedup_hybrid": speedup_hybrid,
                "efficiency_mpi": efficiency_parallel,
                "efficiency_hybrid": efficiency_hybrid,
                "processes": self.size,
                "dimension": self.n
            }
        
        return None
    
    def _plot_performance_comparison(self, seq_time, parallel_time, hybrid_time, speedup_parallel, speedup_hybrid):
        """
        Plot performance comparison between sequential and parallel implementations.
        
        Args:
            seq_time (float): Sequential construction time.
            parallel_time (float): Parallel construction time (MPI only).
            hybrid_time (float): Parallel construction time (MPI+OpenMP).
            speedup_parallel (float): Speedup for MPI only.
            speedup_hybrid (float): Speedup for MPI+OpenMP.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot execution times
        labels = ['Sequential', 'MPI Only', 'MPI+OpenMP']
        times = [seq_time, parallel_time, hybrid_time]
        
        bars = ax1.bar(labels, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height:.4f}s", ha='center', va='bottom')
        
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        
        # Plot speedup
        labels = ['MPI Only', 'MPI+OpenMP']
        speedups = [speedup_parallel, speedup_hybrid]
        
        bars = ax2.bar(labels, speedups, color=['#ff7f0e', '#2ca02c'])
        
        # Add ideal speedup line
        ax2.axhline(y=self.size, color='r', linestyle='--', alpha=0.7, label=f'Ideal Speedup ({self.size}x)')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height:.2f}x", ha='center', va='bottom')
        
        ax2.set_ylabel('Speedup')
        ax2.set_title('Speedup Comparison')
        ax2.legend()
        
        plt.suptitle(f'Performance Comparison for B{self.n} with {self.size} processes')
        plt.tight_layout()
        plt.savefig(f"performance_comparison_B{self.n}_P{self.size}.png", dpi=300)
        plt.show()


def run_sequential_for_comparison(n):
    """
    Run sequential algorithm to obtain baseline metrics for comparison.
    
    Args:
        n (int): Dimension of the bubble-sort network.
        
    Returns:
        dict: Performance metrics from sequential implementation.
    """
    seq_bn = BubbleSortNetwork(n)
    
    # Measure time to construct all trees
    start_time = time.time()
    seq_bn.construct_all_ists()
    construction_time = time.time() - start_time
    
    return {
        "dimension": n,
        "vertices": len(seq_bn.vertices),
        "trees": n-1,
        "construction_time": construction_time,
    }


def main():
    """
    Main function to demonstrate the parallel algorithm.
    """
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("Bubble-Sort Network IST Construction Algorithm (Parallel Version)")
        print("=" * 70)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Construct independent spanning trees in bubble-sort networks in parallel.')
    parser.add_argument('--dimension', '-d', type=int, default=4, help='Dimension of the bubble-sort network (default: 4)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize trees (only in root process)')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare with sequential implementation')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file for performance data')
    
    args = parser.parse_args()
    
    n = args.dimension
    if n < 3 and rank == 0:
        print("Error: Dimension must be at least 3.")
        sys.exit(1)
    
    # Create parallel bubble-sort network
    p_bn = ParallelBubbleSortNetwork(n, comm)
    
    # Run the parallel algorithm
    trees = p_bn.construct_all_ists_hybrid()
    
    # Verify independence in root process
    if rank == 0 and trees:
        p_bn.verify_independence(trees)
    
    # Visualize trees if requested (only in root process)
    if rank == 0 and args.visualize and trees:
        p_bn.visualize_all_trees(trees)
    
    # Compare with sequential implementation if requested
    comparison_results = None
    if args.compare:
        if rank == 0:
            # Run sequential algorithm for comparison
            print("\nRunning sequential algorithm for comparison...")
            seq_metrics = run_sequential_for_comparison(n)
            
            # Compare with parallel performance
            comparison_results = p_bn.compare_with_sequential(seq_metrics)
        else:
            # Non-root processes still need to run performance analysis
            p_bn.performance_analysis_parallel()
    else:
        # Just run performance analysis
        p_bn.performance_analysis_parallel()
    
    # Save performance data if output file is specified
    if rank == 0 and args.output and comparison_results:
        with open(args.output, 'w') as f:
            json.dump(comparison_results, f, indent=4)
        print(f"Performance data saved to {args.output}")


if __name__ == "__main__":
    main()