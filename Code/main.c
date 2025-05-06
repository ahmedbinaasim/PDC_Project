#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <metis.h>
#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>
#include <sys/stat.h>
#include <sys/time.h>

#define MAX_N 12
#define MIN_N 3
#define FACTORIAL(n) ((int)tgamma((n)+1))

typedef struct {
    int *perm;
    int *inverse;
    int r_value;
} VertexData;

typedef struct {
    idx_t *xadj;
    idx_t *adjncy;
    int num_vertices;
    int num_edges;
} GraphCSR;

double get_wall_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

void initialize_parallel_environment(int n) {
    omp_set_num_threads(n - 1);
    mkdir("output", 0755);
    mkdir("output_IST", 0755);
}

bool is_adjacent_swap(const int *perm1, const int *perm2, int n) {
    int diff_count = 0, pos1 = -1, pos2 = -1;
    for (int i = 0; i < n; i++) {
        if (perm1[i] != perm2[i]) {
            if (diff_count < 2) {
                if (diff_count == 0) pos1 = i;
                else pos2 = i;
            }
            diff_count++;
        }
    }
    return diff_count == 2 && pos2 == pos1 + 1 &&
           perm1[pos1] == perm2[pos2] && perm1[pos2] == perm2[pos1];
}

void generate_permutations(int **perms, int n, int *count) {
    int *current = malloc(n * sizeof(int));
    bool *used = calloc(n, sizeof(bool));
    int *stack = malloc(n * sizeof(int));
    int top = 0;

    stack[top++] = -1;
    while (top > 0) {
        int i = stack[--top];
        if (i != -1) used[i] = false;

        for (i++; i < n && used[i]; i++);
        if (i < n) {
            stack[top++] = i;
            used[i] = true;
            current[top - 1] = i + 1;

            if (top == n) {
                memcpy(perms[(*count)++], current, n * sizeof(int));
            } else {
                stack[top++] = -1;
            }
        }
    }
    free(current);
    free(used);
    free(stack);
}

void compute_vertex_data(int **perms, VertexData *data, int n, int num_perms) {
    #pragma omp parallel for
    for (int i = 0; i < num_perms; i++) {
        data[i].perm = perms[i];
        data[i].inverse = malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            data[i].inverse[perms[i][j] - 1] = j + 1;
        }
        data[i].r_value = -1;
        for (int j = n - 1; j >= 0; j--) {
            if (perms[i][j] != j + 1) {
                data[i].r_value = j + 1;
                break;
            }
        }
    }
}

bool is_identity_perm(const int *perm, int n) {
    for (int i = 0; i < n; i++) {
        if (perm[i] != i + 1) return false;
    }
    return true;
}

void swap_vertices(const int *v, int n, int x, int *result) {
    memcpy(result, v, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        if (v[i] == x && i < n - 1) {
            int temp = result[i];
            result[i] = result[i + 1];
            result[i + 1] = temp;
            break;
        }
    }
}

bool swap_results_in_identity(const int *v, int n, int x) {
    int *temp = malloc(n * sizeof(int));
    swap_vertices(v, n, x, temp);
    bool result = is_identity_perm(temp, n);
    free(temp);
    return result;
}

void find_position(const VertexData *v, int n, int t, int *result) {
    if (t == 2 && swap_results_in_identity(v->perm, n, t)) {
        swap_vertices(v->perm, n, t - 1, result);
    } else if (v->perm[n - 2] == t || v->perm[n - 2] == n - 1) {
        swap_vertices(v->perm, n, v->r_value, result);
    } else {
        swap_vertices(v->perm, n, t, result);
    }
}

void compute_parent(const VertexData *v, int n, int t, int *parent_perm) {
    if (v->perm[n - 1] == n) {
        if (t != n - 1) {
            find_position(v, n, t, parent_perm);
        } else {
            swap_vertices(v->perm, n, v->perm[n - 2], parent_perm);
        }
    } else if (v->perm[n - 1] == n - 1 && v->perm[n - 2] == n &&
               !swap_results_in_identity(v->perm, n, n)) {
        swap_vertices(v->perm, n, t == 1 ? n : t - 1, parent_perm);
    } else {
        swap_vertices(v->perm, n, v->perm[n - 1] == t ? n : t, parent_perm);
    }
}

int permutation_index(const int *perm, int **perms, int num_perms, int n) {
    for (int i = 0; i < num_perms; i++) {
        bool match = true;
        for (int j = 0; j < n; j++) {
            if (perms[i][j] != perm[j]) {
                match = false;
                break;
            }
        }
        if (match) return i;
    }
    return -1;
}

void build_csr_graph(int **perms, int n, int num_perms, GraphCSR *graph) {
    int *edge_counts = calloc(num_perms, sizeof(int));
    for (int i = 0; i < num_perms; i++) {
        for (int j = i + 1; j < num_perms; j++) {
            if (is_adjacent_swap(perms[i], perms[j], n)) {
                edge_counts[i]++;
                edge_counts[j]++;
            }
        }
    }

    graph->num_vertices = num_perms;
    graph->num_edges = 0;
    for (int i = 0; i < num_perms; i++) {
        graph->num_edges += edge_counts[i];
    }

    graph->xadj = malloc((num_perms + 1) * sizeof(idx_t));
    graph->adjncy = malloc(graph->num_edges * sizeof(idx_t));
    graph->xadj[0] = 0;
    for (int i = 0; i < num_perms; i++) {
        graph->xadj[i + 1] = graph->xadj[i] + edge_counts[i];
    }

    int *positions = calloc(num_perms, sizeof(int));
    for (int i = 0; i < num_perms; i++) {
        for (int j = 0; j < num_perms; j++) {
            if (i != j && is_adjacent_swap(perms[i], perms[j], n)) {
                graph->adjncy[graph->xadj[i] + positions[i]++] = j;
            }
        }
    }
    free(edge_counts);
    free(positions);
}

void partition_graph(GraphCSR *graph, int n_parts, idx_t *part, int rank, int size) {
    if (graph->num_vertices < size || n_parts <= 1) {
        // For small graphs or single process, assign all vertices to rank 0
        for (int i = 0; i < graph->num_vertices; i++) {
            part[i] = 0;
        }
        if (rank == 0) {
            printf("Bypassing METIS: graph too small or single process (vertices: %d, processes: %d)\n",
                   graph->num_vertices, size);
        }
        return;
    }

    idx_t ncon = 1, objval, nparts = n_parts;
    real_t *tpwgts = malloc(ncon * nparts * sizeof(real_t));
    real_t ubvec = 1.001;
    for (int i = 0; i < nparts; i++) {
        tpwgts[i] = 1.0 / nparts;
    }

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_UFACTOR] = 1;
    options[METIS_OPTION_NCUTS] = 10;
    options[METIS_OPTION_SEED] = 123;
    options[METIS_OPTION_MINCONN] = 1;
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;

    if (rank == 0) {
        int ret = METIS_PartGraphKway(&graph->num_vertices, &ncon, graph->xadj,
                                      graph->adjncy, NULL, NULL, NULL, &nparts,
                                      tpwgts, &ubvec, options, &objval, part);
        if (ret != METIS_OK) {
            printf("METIS failed: %d\n", ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("METIS edge-cut: %d\n", objval);
    }
    MPI_Bcast(part, graph->num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    free(tpwgts);
}

void visualize_graph(int **perms, int n, int num_perms, GraphCSR *graph, int rank) {
    if (rank != 0 || n > 4) return;

    GVC_t *gvc = gvContext();
    Agraph_t *g = agopen("BubbleSortGraph", Agundirected, NULL);
    agattr(g, AGRAPH, "label", "Bubble-Sort Network B_n");

    Agnode_t **nodes = malloc(num_perms * sizeof(Agnode_t *));
    for (int i = 0; i < num_perms; i++) {
        char id[32], label[64];
        snprintf(id, sizeof(id), "node%d", i);
        nodes[i] = agnode(g, id, 1);
        snprintf(label, sizeof(label), "[%d", perms[i][0]);
        for (int j = 1; j < n; j++) {
            snprintf(label + strlen(label), sizeof(label) - strlen(label), ",%d", perms[i][j]);
        }
        strcat(label, "]");
        agsafeset(nodes[i], "label", label, "");
        if (is_identity_perm(perms[i], n)) {
            agsafeset(nodes[i], "color", "blue", "");
            agsafeset(nodes[i], "style", "filled", "");
            agsafeset(nodes[i], "fillcolor", "lightblue", "");
        }
    }

    for (int i = 0; i < num_perms; i++) {
        for (idx_t j = graph->xadj[i]; j < graph->xadj[i + 1]; j++) {
            if (i < graph->adjncy[j]) {
                agedge(g, nodes[i], nodes[graph->adjncy[j]], NULL, 1);
            }
        }
    }

    char filename[128];
    snprintf(filename, sizeof(filename), "output/graph_B%d.png", n);
    gvLayout(gvc, g, "neato");
    gvRenderFilename(gvc, g, "png", filename);
    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);
    free(nodes);
}

void visualize_ist(int t, int **perms, int *parents, int n, int num_perms, int identity_idx, int rank) {
    if (rank != 0 || n > 4) return;

    GVC_t *gvc = gvContext();
    Agraph_t *g = agopen("IST", Agdirected, NULL);
    char title[64];
    snprintf(title, sizeof(title), "Independent Spanning Tree T%d", t + 1);
    agattr(g, AGRAPH, "label", title);

    Agnode_t **nodes = malloc(num_perms * sizeof(Agnode_t *));
    for (int i = 0; i < num_perms; i++) {
        char id[32], label[64];
        snprintf(id, sizeof(id), "node%d", i);
        nodes[i] = agnode(g, id, 1);
        snprintf(label, sizeof(label), "[%d", perms[i][0]);
        for (int j = 1; j < n; j++) {
            snprintf(label + strlen(label), sizeof(label) - strlen(label), ",%d", perms[i][j]);
        }
        strcat(label, "]");
        agsafeset(nodes[i], "label", label, "");
        if (i == identity_idx) {
            agsafeset(nodes[i], "color", "red", "");
            agsafeset(nodes[i], "style", "filled", "");
            agsafeset(nodes[i], "fillcolor", "lightpink", "");
        }
    }

    for (int i = 0; i < num_perms; i++) {
        if (i != identity_idx && parents[i] >= 0) {
            agedge(g, nodes[parents[i]], nodes[i], NULL, 1);
        }
    }

    char filename[128];
    snprintf(filename, sizeof(filename), "output_IST/tree_T%d_B%d.png", t + 1, n);
    gvLayout(gvc, g, "dot");
    gvRenderFilename(gvc, g, "png", filename);
    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);
    free(nodes);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = get_wall_time();

    int n = 4; // Default to B4
    if (argc > 1) n = atoi(argv[1]);
    if (n < MIN_N || n > MAX_N) {
        if (rank == 0) printf("N must be between %d and %d\n", MIN_N, MAX_N);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Running B%d with %d process(es) (Vertices: %d)\n", n, size, FACTORIAL(n));
    }

    initialize_parallel_environment(n);
    int num_perms = FACTORIAL(n);
    int **perms = malloc(num_perms * sizeof(int *));
    for (int i = 0; i < num_perms; i++) {
        perms[i] = malloc(n * sizeof(int));
    }
    int count = 0;
    generate_permutations(perms, n, &count);

    VertexData *vertices = malloc(num_perms * sizeof(VertexData));
    compute_vertex_data(perms, vertices, n, num_perms);

    int identity_idx = -1;
    for (int i = 0; i < num_perms; i++) {
        if (is_identity_perm(perms[i], n)) {
            identity_idx = i;
            break;
        }
    }

    GraphCSR graph = {0};
    build_csr_graph(perms, n, num_perms, &graph);
    idx_t *part = malloc(num_perms * sizeof(idx_t));
    partition_graph(&graph, size, part, rank, size);

    int *local_vertices = malloc(num_perms * sizeof(int));
    int local_count = 0;
    for (int i = 0; i < num_perms; i++) {
        if (part[i] == rank) {
            local_vertices[local_count++] = i;
        }
    }

    visualize_graph(perms, n, num_perms, &graph, rank);

    int **parents = malloc((n - 1) * sizeof(int *));
    for (int t = 0; t < n - 1; t++) {
        parents[t] = malloc(num_perms * sizeof(int));
        memset(parents[t], -1, num_perms * sizeof(int));
    }

    #pragma omp parallel for collapse(2)
    for (int t = 1; t <= n - 1; t++) {
        for (int i = 0; i < local_count; i++) {
            int idx = local_vertices[i];
            if (idx == identity_idx) continue;
            int parent_perm[MAX_N];
            compute_parent(&vertices[idx], n, t, parent_perm);
            parents[t - 1][idx] = permutation_index(parent_perm, perms, num_perms, n);
        }
    }

    if (rank == 0) {
        for (int t = 0; t < n - 1; t++) {
            parents[t][identity_idx] = identity_idx;
        }
    }

    int **global_parents = NULL;
    if (rank == 0) {
        global_parents = malloc((n - 1) * sizeof(int *));
        for (int t = 0; t < n - 1; t++) {
            global_parents[t] = malloc(num_perms * sizeof(int));
            memset(global_parents[t], -1, num_perms * sizeof(int));
        }
    }

    for (int t = 0; t < n - 1; t++) {
        if (rank == 0) {
            memcpy(global_parents[t], parents[t], num_perms * sizeof(int));
            for (int src = 1; src < size; src++) {
                int *buffer = malloc(num_perms * sizeof(int));
                MPI_Recv(buffer, num_perms, MPI_INT, src, t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < num_perms; i++) {
                    if (buffer[i] != -1) {
                        global_parents[t][i] = buffer[i];
                    }
                }
                free(buffer);
            }
        } else {
            MPI_Send(parents[t], num_perms, MPI_INT, 0, t, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        for (int t = 0; t < n - 1; t++) {
            visualize_ist(t, perms, global_parents[t], n, num_perms, identity_idx, rank);
            // printf("Tree T%d:\n", t + 1);
            // for (int i = 0; i < num_perms; i++) {
            //     printf("  Vertex [%d", perms[i][0]);
            //     for (int j = 1; j < n; j++) printf(",%d", perms[i][j]);
            //     printf("] -> Parent ");
            //     if (global_parents[t][i] >= 0) {
            //         printf("[%d", perms[global_parents[t][i]][0]);
            //         for (int j = 1; j < n; j++) printf(",%d", perms[global_parents[t][i]][j]);
            //         printf("]");
            //     } else {
            //         printf("[none]");
            //     }
            //     printf("\n");
            // }
        }
    }

    for (int i = 0; i < num_perms; i++) {
        free(perms[i]);
        free(vertices[i].inverse);
    }
    free(perms);
    free(vertices);
    free(local_vertices);
    free(part);
    free(graph.xadj);
    free(graph.adjncy);
    for (int t = 0; t < n - 1; t++) {
        free(parents[t]);
        if (rank == 0) free(global_parents[t]);
    }
    free(parents);
    if (rank == 0) free(global_parents);

    double end_time = get_wall_time();
    if (rank == 0) {
        printf("Execution time for B%d with %d process(es): %.4f seconds\n",
               n, size, end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}