#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"

#define g(x, y) (g[y*n+x]) 

/**
 * Pr(x) = (1-d)/n + d*sum_{n in g(n,x)}(Pr(n)/(outdegree n))
 * Runs 1 iteration of pagerank
 * Returns 1 if done, 0 otherwise
 */
int run_iteration(int n, double d, int* restrict g, double* restrict w) 
{
    int* restrict wnew = (int*) calloc(n, sizeof(double));
    int done = 1;
    #pragma omp parallel for shared(g, w, wnew) reduction(&& : done)
    for (int i=0; i<n; ++i) {
        double sum = 0.0;
        for (int j=0; j<n; ++j) {
            //find edges pointing toward i
            if (g(j,i)) { 
                //count out degree of j
                int jDegree = 0;
                for (int k=0; k<n; ++k) {
                    jDegree += g(j,k);
                }
                sum += w[j]/(double)jDegree;
            }
        }
        wnew[i] = ((1.0 - d)/n) + (d*sum);
        done = wnew[i] == w[i];
    }
    printf("Iteration Happened\n");
    memcpy(w, wnew, n * sizeof(double));
    free(wnew);
    return done;
}

/**
 *
 */

void pagerank(int n, double d, int* restrict g, double* restrict w)
{
    for (int done = 0; !done; ) {
        done = run_iteration(n, d, g, w);
    }
}

/**
 * # The random graph model
 *
 * Of course, we need to run the shortest path algorithm on something!
 * For the sake of keeping things interesting, let's use a simple random graph
 * model to generate the input data.  The $G(n,p)$ model simply includes each
 * possible edge with probability $p$, drops it otherwise -- doesn't get much
 * simpler than that.  We use a thread-safe version of the Mersenne twister
 * random number generator in lieu of coin flips.
 */

int* gen_graph(int n, double p)
{
    int* g = calloc(n*n, sizeof(int));
    struct mt19937p state;
    sgenrand(10302011UL, &state);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i)
            g(i, j) = (genrand(&state) < p);
        g(j, j) = 0; //no self edges
    }
    return g;
}

void write_matrix(const char* fname, int n, int* g)
{
    FILE* fp = fopen(fname, "w+");
    if (fp == NULL) {
        fprintf(stderr, "Could not open output file: %s\n", fname);
        exit(-1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) 
            fprintf(fp, "%d ", g(i,j));
        fprintf(fp, "\n");
    }
    fclose(fp);
}


void write_weights(const char* fname, int n, double* w)
{
    FILE* fp = fopen(fname, "w+");
    if (fp == NULL) {
        fprintf(stderr, "Could not open output file: %s\n", fname);
        exit(-1);
    }
    for (int i = 0; i < n; ++i) {
        fprintf(fp, "%g ", w[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}

double checksum(const double* restrict w, int n) {
    double sum = 0.0;
    for (int i=0; i<n; ++i) {
        sum += w[i];
    }
    return sum;
}

/**
 * # The `main` event
 */

const char* usage =
    "pagerank.x -- Compute pagerank on a random graph\n"
    "Flags:\n"
    "  - n -- number of nodes (200)\n"
    "  - p -- probability of including edges (0.05)\n"
    "  - d -- probability that a user follows a link (0.85)\n"
    "  - i -- file name where adjacency matrix should be stored (none)\n"
    "  - o -- file name where output weights should be stored (none)\n";

int main(int argc, char** argv)
{
    int n    = 200;            // Number of nodes
    double p = 0.05;           // Edge probability
    double d = 0.85;           // Probability a link is followed
    const char* ifname = NULL; // Adjacency matrix file name
    const char* ofname = NULL; // Distance matrix file name

    // Option processing
    extern char* optarg;
    const char* optstring = "hn:d:p:o:i:";
    int c;
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
        case 'h':
            fprintf(stderr, "%s", usage);
            return -1;
        case 'n': n = atoi(optarg); break;
        case 'p': p = atof(optarg); break;
        case 'd': d = atof(optarg); break;
        case 'o': ofname = optarg;  break;
        case 'i': ifname = optarg;  break;
        }
    }

    // Graph generation + output
    int* g = gen_graph(n, p);
    if (ifname)
        write_matrix(ifname, n, g);

    // Generate initial weights
    double* w = calloc(n, sizeof(double));
    for (int i = 0; i < n; ++i) {
        w[i] = 1.0/(double)n;
    }

    // Time the pagerank code
    double t0 = omp_get_wtime();
    pagerank(n, d, g, w);
    double t1 = omp_get_wtime();

    printf("== OpenMP with %d threads\n", omp_get_max_threads());
    printf("n:     %d\n", n);
    printf("p:     %g\n", p);
    printf("d:     %g\n", d);
    printf("Time:  %g\n", t1-t0);
    printf("Check: %g\n", checksum(w, n));

    // Generate output file
    if (ofname)
        write_weights(ofname, n, w);

    // Clean up
    free(g);
    free(w);
    return 0;
}
