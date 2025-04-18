# Implementation Code

This directory contains the source code for our implementation of the parallel algorithm for constructing multiple independent spanning trees in bubble-sort networks.

## Project Structure

- **src/**: Source code files
  - **sequential/**: Sequential implementation of the algorithm
  - **parallel/**: Parallel implementation using MPI and OpenMP
  - **utils/**: Utility functions and data structures
- **include/**: Header files
- **tests/**: Test files and test datasets
- **scripts/**: Scripts for running experiments and collecting results
- **results/**: Performance results and analysis

## Implementation Details

Our implementation focuses on three main aspects:

1. **Bubble-Sort Network Construction**: Code to represent and generate bubble-sort networks of different dimensions
2. **IST Construction Algorithm**: Implementation of the non-recursive algorithm for parent determination
3. **Parallelization**: Code using MPI and OpenMP for parallel execution

### Key Components:

- **BubbleSortNetwork**: Class representing bubble-sort networks
- **ISTConstructor**: Class implementing the IST construction algorithm
- **GraphPartitioner**: Class using METIS for graph partitioning
- **Performance Monitor**: Utilities to measure and record performance metrics

## Building and Running

Prerequisites:
- MPI implementation (OpenMPI recommended)
- OpenMP-compatible compiler
- METIS library

Build instructions:
```bash
# Instructions will be added as implementation progresses
```

Running examples:
```bash
# Examples will be added as implementation progresses
```

## Team Members
- Abeerah Aamir (22i-0758)
- Ahmed Bin Asim (22i-0949)
- Section CS-C