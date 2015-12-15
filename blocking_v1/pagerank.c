#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"

#define g(x, y) (g[y*n+x]) 

void run_block(int n, double d, int* restrict g, double* restrict w, double* restrict wnew, int* restrict degree, int start, int count) 
{
    double* wlocalnew = (double*) calloc(count, sizeof(double));
    memcpy(wlocalnew, w + start, count * sizeof(double));

    double* wlocal = (double*) calloc(count, sizeof(double));
    memcpy(wlocal, wlocalnew, count * sizeof(double));
    

    int done = 0;
    while (!done) {
        done = 1;
        memcpy(wlocal, wlocalnew, count * sizeof(double));

        for (int i=0; i<count; ++i) {
            double sum = 0.0;
            //do before the block
            for (int j=0; j<start; ++j) {
                //find edges pointing toward i
                if (g(j,i+start)) { 
                    //count out degree of j
                    sum += w[j]/(double)degree[j];
                }
            }

            // do the block
            for (int j=start; j<count; ++j) {
                //find edges pointing toward i
                if (g(j,i+start)) { 
                    //count out degree of j
                    sum += wlocal[j]/(double)degree[j];
                }
            }

            // do after the block
            for (int j=start+count; j<n; ++j) {
                //find edges pointing toward i
                if (g(j,i+start)) { 
                    //count out degree of j
                    sum += w[j]/(double)degree[j];
                }
            }

            wlocalnew[i] = ((1.0 - d)/(double)n) + (d*sum);
            done = done && (wlocalnew[i] == wlocal[i]);
        }

    }
    memcpy(wnew + start, wlocalnew, count * sizeof(double));
}


/**
 * Pr(x) = (1-d)/n + d*sum_{n in g(n,x)}(Pr(n)/(outdegree n))
 * Runs 1 iteration of pagerank
 * Returns 1 if done, 0 otherwise
 */
int run_iteration(int n, double d, int* restrict g, double* restrict w, double* restrict wnew, int* restrict degree) 
{
    #pragma omp parallel shared(w, wnew)
    {
        int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
        int start = (n/num_threads) * this_thread;
        int count;
        if (this_thread == num_threads - 1) {
            count = n - start;
        } else {
            count = ((n/num_threads) * (this_thread + 1)) - start;
        }
        run_block(n, d, g, w, wnew, degree, start, count);
    }
    int done = 1;
    for(int i=0; i<n; i++){
        done = done && w[i] == wnew[i];
        w[i] = wnew[i];
    }
    return done;
}

/**
 *
 */

int pagerank(int n, double d, int* restrict g, double* restrict w)
{
    int iterations = 0;
    double* restrict wnew = (double*) calloc(n, sizeof(double));
    
    //compute degree of each item prior (if degree = 0, it should be n)
    int* restrict degree = (int*) calloc(n, sizeof(int));
    for (int i=0; i<n; ++i) {
        int count = 0;
        for (int j=0; j<n; ++j) {
            count += g(i,j);
        }

        if (count == 0) {
            count = n;
        }
        degree[i] = count;
    }

    for (int done = 0; !done; ) {
        done = run_iteration(n, d, g, w, wnew, degree);
        iterations++;
    }

    free(wnew);
    free(degree);
    return iterations;
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
    sgenrand(time(NULL), &state);
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
    int iterations = pagerank(n, d, g, w);
    double t1 = omp_get_wtime();

    //openmp, cores, time, n, iterations, p, d, checksum
    printf("openmp, %d, %g, %d, %d, %g, %g, %g\n", 
           omp_get_max_threads(),
           (t1-t0),
           n,
           iterations,
           p,
           d,
           checksum(w, n));

    // Generate output file
    if (ofname)
        write_weights(ofname, n, w);

    // Clean up
    free(g);
    free(w);
    return 0;
}