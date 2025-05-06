#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing script for performance analysis of sequential and parallel implementations
of the algorithm for constructing independent spanning trees in bubble-sort networks.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import subprocess
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from datetime import datetime

def run_sequential_test(dimension, runs=3):
    """
    Run the sequential implementation multiple times and return average performance.
    
    Args:
        dimension (int): Dimension of the bubble-sort network.
        runs (int): Number of times to run the test for averaging.
        
    Returns:
        dict: Average performance metrics.
    """
    print(f"Running sequential tests for B{dimension} ({runs} runs)...")
    
    results = []
    for i in tqdm(range(runs)):
        # Run the sequential script and capture output
        cmd = f"python sequential.py --dimension {dimension} --performance"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Parse the output to extract performance metrics
        output = stdout.decode('utf-8')
        
        # Extract construction time from output
        try:
            construction_time = float(output.split("Construction Time:")[1].split("seconds")[0].strip())
            verification_time = float(output.split("Verification Time:")[1].split("seconds")[0].strip())
            total_time = float(output.split("Total Time:")[1].split("seconds")[0].strip())
            
            results.append({
                "run": i+1,
                "dimension": dimension,
                "construction_time": construction_time,
                "verification_time": verification_time,
                "total_time": total_time
            })
        except:
            print(f"Error parsing output for run {i+1}. Skipping.")
            print("Output:", output)
            print("Error:", stderr.decode('utf-8'))
    
    # Calculate average metrics
    if results:
        avg_construction_time = sum(r["construction_time"] for r in results) / len(results)
        avg_verification_time = sum(r["verification_time"] for r in results) / len(results)
        avg_total_time = sum(r["total_time"] for r in results) / len(results)
        
        return {
            "dimension": dimension,
            "avg_construction_time": avg_construction_time,
            "avg_verification_time": avg_verification_time,
            "avg_total_time": avg_total_time,
            "runs": len(results),
            "raw_results": results
        }
    
    return None

def run_parallel_test(dimension, processes, runs=3):
    """
    Run the parallel implementation multiple times and return average performance.
    
    Args:
        dimension (int): Dimension of the bubble-sort network.
        processes (int): Number of MPI processes to use.
        runs (int): Number of times to run the test for averaging.
        
    Returns:
        dict: Average performance metrics.
    """
    print(f"Running parallel tests for B{dimension} with {processes} processes ({runs} runs)...")
    
    results = []
    output_file = f"parallel_performance_B{dimension}_P{processes}.json"
    
    for i in tqdm(range(runs)):
        # Run the parallel script with MPI and capture output
        cmd = f"mpiexec -n {processes} python parallel.py --dimension {dimension} --compare --output {output_file}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Check if output file was created
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    run_data = json.load(f)
                run_data["run"] = i+1
                results.append(run_data)
            except:
                print(f"Error reading output file for run {i+1}. Skipping.")
        else:
            print(f"Output file not created for run {i+1}. Skipping.")
            print("Output:", stdout.decode('utf-8'))
            print("Error:", stderr.decode('utf-8'))
    
    # Calculate average metrics
    if results:
        avg_sequential_time = sum(r["sequential_time"] for r in results) / len(results)
        avg_parallel_time_mpi = sum(r["parallel_time_mpi"] for r in results) / len(results)
        avg_parallel_time_hybrid = sum(r["parallel_time_hybrid"] for r in results) / len(results)
        avg_speedup_mpi = sum(r["speedup_mpi"] for r in results) / len(results)
        avg_speedup_hybrid = sum(r["speedup_hybrid"] for r in results) / len(results)
        avg_efficiency_mpi = sum(r["efficiency_mpi"] for r in results) / len(results)
        avg_efficiency_hybrid = sum(r["efficiency_hybrid"] for r in results) / len(results)
        
        return {
            "dimension": dimension,
            "processes": processes,
            "avg_sequential_time": avg_sequential_time,
            "avg_parallel_time_mpi": avg_parallel_time_mpi,
            "avg_parallel_time_hybrid": avg_parallel_time_hybrid,
            "avg_speedup_mpi": avg_speedup_mpi,
            "avg_speedup_hybrid": avg_speedup_hybrid,
            "avg_efficiency_mpi": avg_efficiency_mpi,
            "avg_efficiency_hybrid": avg_efficiency_hybrid,
            "runs": len(results),
            "raw_results": results
        }
    
    return None

def run_scaling_tests(dimensions, processes_list, runs=3):
    """
    Run a series of tests to evaluate strong and weak scaling.
    
    Args:
        dimensions (list): List of bubble-sort network dimensions to test.
        processes_list (list): List of process counts to test.
        runs (int): Number of runs per configuration.
        
    Returns:
        dict: Scaling test results.
    """
    results = {
        "strong_scaling": [],  # Fixed problem size, varying process count
        "weak_scaling": []     # Problem size proportional to process count
    }
    
    # Strong scaling tests (fixed problem size, varying process count)
    for dim in dimensions:
        print(f"\nStrong Scaling Tests for B{dim}:")
        dim_results = []
        
        # Run sequential version first
        seq_result = run_sequential_test(dim, runs)
        if seq_result:
            dim_results.append({
                "dimension": dim,
                "processes": 1,
                "time": seq_result["avg_construction_time"],
                "speedup": 1.0,
                "efficiency": 1.0
            })
        
        # Run parallel versions with different process counts
        for p in processes_list:
            if p > 1:  # Skip p=1 since we already ran sequential version
                par_result = run_parallel_test(dim, p, runs)
                if par_result:
                    dim_results.append({
                        "dimension": dim,
                        "processes": p,
                        "time": par_result["avg_parallel_time_hybrid"],
                        "speedup": par_result["avg_speedup_hybrid"],
                        "efficiency": par_result["avg_efficiency_hybrid"]
                    })
        
        results["strong_scaling"].extend(dim_results)
    
    # Weak scaling tests (problem size increases with process count)
    # For bubble-sort networks, dimension is related to problem size
    # We'll use B3 for 1 process, B4 for 2-4 processes, B5 for 5-6 processes, etc.
    weak_dims = []
    for p in processes_list:
        if p == 1:
            weak_dims.append(3)
        elif p <= 4:
            weak_dims.append(4)
        elif p <= 6:
            weak_dims.append(5)
        else:
            weak_dims.append(6)  # Cap at B6 due to factorial growth
    
    print("\nWeak Scaling Tests:")
    for p, dim in zip(processes_list, weak_dims):
        if p == 1:
            seq_result = run_sequential_test(dim, runs)
            if seq_result:
                results["weak_scaling"].append({
                    "dimension": dim,
                    "processes": p,
                    "time": seq_result["avg_construction_time"],
                    "vertices": np.math.factorial(dim),
                    "time_per_vertex": seq_result["avg_construction_time"] / np.math.factorial(dim)
                })
        else:
            par_result = run_parallel_test(dim, p, runs)
            if par_result:
                results["weak_scaling"].append({
                    "dimension": dim,
                    "processes": p,
                    "time": par_result["avg_parallel_time_hybrid"],
                    "vertices": np.math.factorial(dim),
                    "time_per_vertex": par_result["avg_parallel_time_hybrid"] / np.math.factorial(dim)
                })
    
    return results

def visualize_strong_scaling(results):
    """
    Visualize strong scaling results.
    
    Args:
        results (list): List of strong scaling test results.
    """
    if not results:
        print("No strong scaling results to visualize.")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Group by dimension
    dimensions = df["dimension"].unique()
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for dim in dimensions:
        dim_data = df[df["dimension"] == dim]
        
        # Execution time vs. processes
        axes[0, 0].plot(dim_data["processes"], dim_data["time"], marker='o', label=f'B{dim}')
        
        # Speedup vs. processes
        axes[0, 1].plot(dim_data["processes"], dim_data["speedup"], marker='o', label=f'B{dim}')
        
        # Efficiency vs. processes
        axes[1, 0].plot(dim_data["processes"], dim_data["efficiency"], marker='o', label=f'B{dim}')
        
        # Log-log plot of time vs. processes
        axes[1, 1].loglog(dim_data["processes"], dim_data["time"], marker='o', label=f'B{dim}')
    
    # Add ideal scaling lines
    max_p = df["processes"].max()
    p_range = np.arange(1, max_p + 1)
    
    # Ideal speedup (y = x)
    axes[0, 1].plot(p_range, p_range, 'k--', alpha=0.5, label='Ideal')
    
    # Ideal efficiency (y = 1)
    axes[1, 0].plot(p_range, [1] * len(p_range), 'k--', alpha=0.5, label='Ideal')
    
    # Set titles and labels
    axes[0, 0].set_title('Execution Time vs. Processes')
    axes[0, 0].set_xlabel('Number of Processes')
    axes[0, 0].set_ylabel('Execution Time (s)')
    
    axes[0, 1].set_title('Speedup vs. Processes')
    axes[0, 1].set_xlabel('Number of Processes')
    axes[0, 1].set_ylabel('Speedup')
    
    axes[1, 0].set_title('Efficiency vs. Processes')
    axes[1, 0].set_xlabel('Number of Processes')
    axes[1, 0].set_ylabel('Efficiency')
    
    axes[1, 1].set_title('Log-Log Plot: Time vs. Processes')
    axes[1, 1].set_xlabel('Number of Processes')
    axes[1, 1].set_ylabel('Execution Time (s)')
    
    # Add legends
    for ax in axes.flatten():
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Strong Scaling Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('strong_scaling_analysis.png', dpi=300)
    plt.show()

def visualize_weak_scaling(results):
    """
    Visualize weak scaling results.
    
    Args:
        results (list): List of weak scaling test results.
    """
    if not results:
        print("No weak scaling results to visualize.")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Execution time vs. processes
    axes[0].plot(df["processes"], df["time"], marker='o', label='Execution Time')
    axes[0].set_title('Weak Scaling: Execution Time vs. Processes')
    axes[0].set_xlabel('Number of Processes')
    axes[0].set_ylabel('Execution Time (s)')
    
    # Add dimension labels
    for i, row in df.iterrows():
        axes[0].annotate(f'B{row["dimension"]}', 
                       (row["processes"], row["time"]),
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='center')
    
    # Time per vertex vs. processes
    axes[1].plot(df["processes"], df["time_per_vertex"], marker='o', label='Time per Vertex')
    axes[1].set_title('Weak Scaling: Time per Vertex vs. Processes')
    axes[1].set_xlabel('Number of Processes')
    axes[1].set_ylabel('Time per Vertex (s)')
    
    # Add ideal scaling line (constant time per vertex)
    ideal_time = df.loc[0, "time_per_vertex"]
    axes[1].plot(df["processes"], [ideal_time] * len(df), 'k--', alpha=0.5, label='Ideal')
    
    # Add labels and grid
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Weak Scaling Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('weak_scaling_analysis.png', dpi=300)
    plt.show()

def visualize_hybrid_vs_mpi(results):
    """
    Visualize comparison between MPI-only and Hybrid (MPI+OpenMP) implementations.
    
    Args:
        results (list): List of strong scaling test results.
    """
    # Filter out sequential results
    par_results = [r for r in results if "avg_parallel_time_mpi" in r and "avg_parallel_time_hybrid" in r]
    
    if not par_results:
        print("No parallel results to visualize.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Group by dimension
    dimensions = set(r["dimension"] for r in par_results)
    
    for dim in dimensions:
        dim_results = [r for r in par_results if r["dimension"] == dim]
        
        # Sort by process count
        dim_results.sort(key=lambda x: x["processes"])
        
        processes = [r["processes"] for r in dim_results]
        mpi_times = [r["avg_parallel_time_mpi"] for r in dim_results]
        hybrid_times = [r["avg_parallel_time_hybrid"] for r in dim_results]
        
        # Execution time comparison
        axes[0].plot(processes, mpi_times, marker='o', linestyle='--', label=f'MPI-only (B{dim})')
        axes[0].plot(processes, hybrid_times, marker='s', label=f'Hybrid (B{dim})')
        
        # Speedup comparison
        mpi_speedups = [r["avg_speedup_mpi"] for r in dim_results]
        hybrid_speedups = [r["avg_speedup_hybrid"] for r in dim_results]
        
        axes[1].plot(processes, mpi_speedups, marker='o', linestyle='--', label=f'MPI-only (B{dim})')
        axes[1].plot(processes, hybrid_speedups, marker='s', label=f'Hybrid (B{dim})')
        
        # Efficiency comparison
        mpi_effs = [r["avg_efficiency_mpi"] for r in dim_results]
        hybrid_effs = [r["avg_efficiency_hybrid"] for r in dim_results]
        
        axes[2].plot(processes, mpi_effs, marker='o', linestyle='--', label=f'MPI-only (B{dim})')
        axes[2].plot(processes, hybrid_effs, marker='s', label=f'Hybrid (B{dim})')
    
    # Add ideal speedup line
    max_p = max(r["processes"] for r in par_results)
    p_range = np.arange(1, max_p + 1)
    axes[1].plot(p_range, p_range, 'k--', alpha=0.5, label='Ideal')
    axes[2].plot(p_range, [1] * len(p_range), 'k--', alpha=0.5, label='Ideal')
    
    # Set titles and labels
    axes[0].set_title('Execution Time: MPI vs. Hybrid')
    axes[0].set_xlabel('Number of Processes')
    axes[0].set_ylabel('Execution Time (s)')
    
    axes[1].set_title('Speedup: MPI vs. Hybrid')
    axes[1].set_xlabel('Number of Processes')
    axes[1].set_ylabel('Speedup')
    
    axes[2].set_title('Efficiency: MPI vs. Hybrid')
    axes[2].set_xlabel('Number of Processes')
    axes[2].set_ylabel('Efficiency')
    
    # Add legends and grid
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('MPI vs. Hybrid Implementation Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('mpi_vs_hybrid_comparison.png', dpi=300)
    plt.show()

def main():
    """
    Main function to run performance tests and visualize results.
    """
    parser = argparse.ArgumentParser(description='Performance testing for IST construction algorithms.')
    parser.add_argument('--dimensions', '-d', type=int, nargs='+', default=[3, 4], 
                        help='Dimensions of bubble-sort networks to test (default: [3, 4])')
    parser.add_argument('--processes', '-p', type=int, nargs='+', default=[1, 2, 4], 
                        help='Number of MPI processes to test (default: [1, 2, 4])')
    parser.add_argument('--runs', '-r', type=int, default=3, 
                        help='Number of runs per configuration for averaging (default: 3)')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output file for performance data (default: None)')
    parser.add_argument('--skip-tests', '-s', action='store_true', 
                        help='Skip running tests and just visualize previous results')
    parser.add_argument('--input', '-i', type=str, default=None, 
                        help='Input file with previous test results for visualization')
    
    args = parser.parse_args()
    
    # Ensure dimensions are valid
    for dim in args.dimensions:
        if dim < 3:
            print(f"Warning: B{dim} is too small. Minimum dimension is 3.")
            args.dimensions.remove(dim)
    
    # Ensure process counts are valid
    for p in args.processes:
        if p < 1:
            print(f"Warning: Invalid process count {p}. Minimum is 1.")
            args.processes.remove(p)
    
    # Set up output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output if args.output else f"performance_results_{timestamp}.json"
    
    results = None
    
    if not args.skip_tests:
        # Run the scaling tests
        results = run_scaling_tests(args.dimensions, args.processes, args.runs)
        
        # Save results to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nPerformance data saved to {output_file}")
    elif args.input:
        # Load results from file
        try:
            with open(args.input, 'r') as f:
                results = json.load(f)
            print(f"Loaded performance data from {args.input}")
        except Exception as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
    else:
        print("Error: Either run tests or provide input file.")
        sys.exit(1)
    
    # Visualize results
    if results:
        if "strong_scaling" in results and results["strong_scaling"]:
            visualize_strong_scaling(results["strong_scaling"])
        
        if "weak_scaling" in results and results["weak_scaling"]:
            visualize_weak_scaling(results["weak_scaling"])
        
        # Collect parallel results for MPI vs. Hybrid comparison
        parallel_results = []
        for dim in args.dimensions:
            for p in args.processes:
                if p > 1:  # Skip sequential
                    for r in results["strong_scaling"]:
                        if r["dimension"] == dim and r["processes"] == p and isinstance(r, dict) and "raw_results" in r:
                            parallel_results.append(r)
        
        if parallel_results:
            visualize_hybrid_vs_mpi(parallel_results)
    
    print("\nPerformance analysis complete.")

if __name__ == "__main__":
    main()