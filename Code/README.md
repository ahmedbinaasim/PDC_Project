Parallel Independent Spanning Trees in Bubble-Sort Networks
This project implements a parallel algorithm for constructing multiple independent spanning trees (ISTs) in bubble-sort networks ( B_n ), as described in a research paper on parallel graph algorithms. The implementation uses MPI for distributed processing, OpenMP for thread-level parallelism, and METIS for graph partitioning. It includes performance analysis capabilities, measuring execution time for graphs ( B_3 ) to ( B_{12} ) with 1, 2, and 4 processes. Visualizations of the bubble-sort graph and ISTs are generated for ( B_3 ) and ( B_4 ).
Overview

Algorithm: Non-recursive parallel construction of ( n-1 ) ISTs in ( B_n ), where ( B_n ) is a bubble-sort network with ( n! ) vertices representing permutations of ( {1, 2, \ldots, n} ).
Features:
Distributed vertex partitioning using METIS (bypassed for small graphs).
Parallel IST computation with OpenMP.
Graph and IST visualization using Graphviz (for ( n \leq 4 )).
Performance timing for 1, 2, and 4 processes.


Output:
Console output of IST parent relationships.
PNG files for graphs (`output/graph_Bn.png`) and ISTs (`output_IST/tree_Tt_Bn.png`) for ( n \leq 4 ).
Execution time for each run.



Prerequisites
The project is designed to run on a Windows Subsystem for Linux (WSL) environment (Ubuntu recommended). Ensure the following dependencies are installed:
sudo apt update
sudo apt install mpich libmpich-dev libmetis-dev libgraphviz-dev build-essential


MPICH: For MPI-based distributed computing.
libmetis-dev: For graph partitioning.
libgraphviz-dev: For visualization with Graphviz.
build-essential: Includes gcc, make, and other tools for compilation.

Verify installations:
mpicc --version
metismex -version
dot -V

Project Structure

main.c: Core implementation with performance timing.
Makefile: Compilation instructions for building the executable.
output/: Directory for bubble-sort graph PNGs.
output_IST/: Directory for IST PNGs.

Compilation

Clone or Copy the Project:Ensure main.c and Makefile are in your project directory (e.g., ~/PDC_Project).

Compile the Code:Run the following command to build the main executable:
make

This compiles main.c with MPI, OpenMP, METIS, and Graphviz dependencies.

Clean Up (Optional):To remove the executable and generated files:
make clean



Running the Program
The program is run manually for each graph size ( B_n ) (where ( n = 3 ) to ( 12 )) with 1, 2, or 4 processes to collect performance data. Each run outputs the execution time and generates visualizations for ( n \leq 4 ).
Run Instructions

Test a Specific Graph:Use the following command to run the program for a specific ( n ) and number of processes:
mpirun -np <processes> ./main <n>


<processes>: Number of MPI processes (1, 2, or 4).
<n>: Graph size (3 to 12).

Examples:

For ( B_3 ) with 1 process:mpirun -np 1 ./main 3


For ( B_4 ) with 4 processes:mpirun -np 4 ./main 4




Performance Analysis:To analyze performance across ( B_3 ) to ( B_{12} ), run the program for each combination of ( n ) and process count:
# B3
mpirun -np 1 ./main 3
mpirun -np 2 ./main 3
mpirun -np 4 ./main 3
# B4
mpirun -np 1 ./main 4
mpirun -np 2 ./main 4
mpirun -np 4 ./main 4
# Repeat for B5 to B12


Output: Each run prints:
Graph size and process count (e.g., Running B3 with 1 process(es) (Vertices: 6)).
METIS edge-cut or bypass message.
IST parent relationships (for rank 0).
Execution time (e.g., Execution time for B3 with 1 process(es): 1.0348 seconds).


Visualizations: For ( n = 3, 4 ), PNGs are saved in output/ and output_IST/.


Collect Performance Data:Record the execution time from each run’s output (line starting with Execution time). Example format:
n,Vertices,Processes,Time
3,6,1,1.0348
3,6,2,0.0359
3,6,4,0.0439
...

Save this data in a text file or table for analysis.


Notes for Large Graphs

B9 to B12: These graphs have large vertex counts (e.g., ( B_{12} ) has 479,001,600 vertices), which may require significant memory and time. Ensure sufficient resources in WSL:free -m


Memory Issues: If runs for ( B_{10} ) to ( B_{12} ) fail, consider limiting to ( B_3 ) to ( B_8 ) or increasing WSL memory allocation.
Execution Time: Large graphs may take hours (e.g., ( B_{12} ) sequential run could exceed 100 hours). Test smaller graphs first.

Troubleshooting

MPI Errors: If MPI fails (e.g., mpirun errors), verify MPICH installation or try OpenMPI:sudo apt install openmpi-bin openmpi-common libopenmpi-dev


METIS Crash: The program bypasses METIS for small graphs (( |V| < \text{processes} )) or single-process runs. If crashes occur, ensure libmetis-dev is installed.
Graphviz Issues: If PNGs are not generated, verify libgraphviz-dev and run:dot -V


WSL Resource Limits: Check WSL memory and CPU allocation in Windows settings. Increase if needed for large graphs.

Output Files

Console: IST parent relationships and execution time for each run.
Visualizations (for ( n \leq 4 )):
output/graph_Bn.png: Bubble-sort graph for ( B_n ).
output_IST/tree_Tt_Bn.png: IST ( T_t ) for ( B_n ).


Access: Files are saved in the project directory, accessible from Windows via the WSL file system.

License
This project is for educational purposes and implements a research paper’s algorithm. Ensure proper citation of the original paper if used academically.
Contact
For issues or questions, contact the project maintainers via email or open an issue in the repository (if hosted).
