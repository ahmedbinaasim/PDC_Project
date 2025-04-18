# Project Documentation

This directory contains documentation for our implementation of the parallel algorithm for constructing multiple independent spanning trees in bubble-sort networks.

## Contents

- **Algorithm Analysis**: Detailed explanation of the algorithm from the paper and our implementation approach
- **Parallelization Strategy**: Documentation of our MPI, OpenMP, and METIS implementation strategies
- **Performance Analysis**: Reports on scalability and efficiency of our implementation
- **Technical Challenges**: Description of challenges faced during implementation and how they were resolved

## Algorithm Overview

The algorithm constructs n-1 independent spanning trees (ISTs) in an n-dimensional bubble-sort network. The key innovation is that it's non-recursive, allowing each vertex to determine its parent in each spanning tree in constant time. This makes it fully parallelizable.

### Key Features:

1. **Constant-time parent determination**: Each vertex uses a set of simple rules to find its parent
2. **Optimal time complexity**: O(n·n!) where n is the dimension and n! is the number of vertices
3. **Height-bounded trees**: The height of the constructed ISTs is at most D(Bn) + n-1

## Parallelization Approach

Our implementation uses three key technologies:

1. **MPI (Message Passing Interface)**: For inter-node communication and work distribution
2. **OpenMP**: For intra-node parallelism to optimize computation on each node
3. **METIS**: For graph partitioning to ensure balanced workload distribution

## Team Members
- Abeerah Aamir (22i-0758)
- Ahmed Bin Asim (22i-0949)
- Section CS-C