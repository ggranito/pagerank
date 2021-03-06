#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"

#define g(x, y) (g[(y)*n+(x)]) 

int run_block(int n, double d, int* restrict g, double* restrict w, double* restrict wnew, int* restrict degree, int start, int count, double* restrict wlocal, int* restrict map) 
{   
    double residual = 0.0;
    for (int i=0; i<count; ++i) {
        double sum = 0.0;
        //do before the block
        for (int j=0; j<start; ++j) {
            //find edges pointing toward i
            if (g(j,map[i+start])) { 
                //count out degree of j
                sum += w[j]/(double)degree[j];
            }
        }

        // do the block
        for (int j=start; j<start+count; ++j) {
            //find edges pointing toward i
            if (g(j,map[i+start])) { 
                //count out degree of j
                sum += wnew[j]/(double)degree[j];
            }
        }

        // do after the block
        for (int j=start+count; j<n; ++j) {
            //find edges pointing toward i
            if (g(j,map[i+start])) { 
                //count out degree of j
                sum += w[j]/(double)degree[j];
            }
        }

        double newVal = ((1.0 - d)/(double)n) + (d*sum);
        residual += fabs(wnew[i+start] - newVal);
        wlocal[i] = newVal;
    }
    return residual < ((double)count)/(1000000.0 * (double)n);
}


/**
 * Pr(x) = (1-d)/n + d*sum_{n in g(n,x)}(Pr(n)/(outdegree n))
 * Runs 1 iteration of pagerank
 * Returns 1 if done, 0 otherwise
 */
int run_iteration(int n, double d, int* restrict g, double* restrict w, double* restrict wnew, int* restrict degree, int* restrict map) 
{
    int iterationDone = 1;
    #pragma omp parallel shared(w, wnew) reduction(&& : iterationDone)
    {
        int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
        int start = (n/num_threads) * this_thread;
        int count;
        if (this_thread == num_threads - 1) {
            count = n - start;
        } else {
            count = ((n/num_threads) * (this_thread + 1)) - start;
        }
        double* wlocal = (double*)calloc(count, sizeof(double));
        memcpy(wlocal, wnew+start, count * sizeof(double));
        int done = 0;
        while (!done) {
            done = run_block(n, d, g, w, wnew, degree, start, count, wlocal, map);            
            memcpy(wnew+start, wlocal, count * sizeof(double));
        }
        free(wlocal);
        #pragma omp barrier
        for(int i=start; i<start+count; i++){
            iterationDone = iterationDone && (fabs(w[i] - wnew[i]) < 1.0/(1000.0 * (double)n));
            w[i] = wnew[i];
        }
    }
    return iterationDone;
}

/**
 *
 */

int pagerank(int n, double d, int* restrict g, double* restrict w)
{
    int iterations = 0;
    double* restrict wnew = (double*) calloc(n, sizeof(double));
    memcpy(wnew, w, n * sizeof(double));
    
    //compute degree of each item prior
    int* restrict degree = (int*) calloc(n, sizeof(int));
    for (int i=0; i<n; ++i) {
        int count = 0;
        for (int j=0; j<n; ++j) {
            count += g(i,j);
        }
        degree[i] = count;
    }

    //group together nodes

    int* restrict map = (int*) calloc(n, sizeof(int));
    int* restrict taken = (int*) calloc(n, sizeof(int));
    for(int i=1;i<n;i++)taken[i]=0;
    map[0] = 0;
    taken[0]=1;
    for (int i = 1; i<n; ++i) {
        int map_i = -1;
        for(int j=0; j<n; ++j) {
            if(!taken[j] && g(i,j)) {
                map_i = j;
                break;
            } else if(!taken[j]) {
                map_i = j;
            }
        }
        map[i] = map_i;
        taken[map_i] = 1;
    }
    free(taken);
    //run
    for (int done = 0; !done; ) {
        done = run_iteration(n, d, g, w, wnew, degree, map);
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
    struct timeval time;
    gettimeofday(&time, NULL);
    sgenrand((unsigned long)time.tv_usec, &state);
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
