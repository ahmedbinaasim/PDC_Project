# Parallel Construction of Independent Spanning Trees in Bubble-Sort Networks

## Project Overview
This repository contains our implementation and analysis of the parallel algorithm for constructing multiple independent spanning trees in bubble-sort networks, as described in the paper by Kao et al. The project is part of the Parallel and Distributed Computing course.

## Team Members
- Abeerah Aamir (22i-0758)
- Ahmed Bin Asim (22i-0949)
- Section CS-C

## Project Structure
This repository is organized into the following directories:

- **[Presentation](./Presentation/)**: Contains our slides and materials for the presentation about the paper and our implementation strategy.
- **[Documentation](./Documentation/)**: Contains project documentation, analysis reports, and implementation details.
- **[Code](./Code/)**: Contains the source code for our implementation using MPI, OpenMP, and METIS.

## Paper Abstract
The use of multiple independent spanning trees (ISTs) for data broadcasting in networks provides advantages including increased fault-tolerance and secure message distribution. This project explores the parallel algorithm proposed by Kao et al. for constructing ISTs in bubble-sort networks. The algorithm enables every vertex to determine its parent in each spanning tree in constant time, allowing for full parallelization.

## Project Objectives
1. Study and understand the parallel algorithm for IST construction in bubble-sort networks
2. Implement the algorithm using MPI, OpenMP, and METIS for graph partitioning
3. Evaluate the scalability and performance of the implementation using various datasets
4. Compare sequential vs. parallel implementations to demonstrate efficiency

## Getting Started
See the README files in each directory for specific information about the contents and how to use them:
- [Documentation README](./Documentation/README.md)
- [Code README](./Code/README.md)