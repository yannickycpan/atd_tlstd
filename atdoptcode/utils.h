#ifndef _UTILS_H
#define _UTILS_H

//Macros for the whole project:
#define printLine printf("Line: %d\n",__LINE__);
#define VAR(x) printf(#x);printf("=%d\n",x);


#define MAX_OPTION_STRING_LENGTH 100
#define MAX_FILENAME_STRING_LENGTH 5000


#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix_double.h>

/***** Each function group in alphabetical order ****/

/* string concat */
char* concat(char *s1, char *s2);

/* Basic functions */
int flatten_index(int * indices, const int * indexrange, const int numindices);

double gettime(struct timeval * tim);

char* itoa(int val, int base);

int int2Binary(gsl_vector * array, int inv, int n);

double log2(double a);

double min(double a, double b);

FILE * open_file(const char * filename, const char * open_argument);


/* Random number generation */
struct rgen_t {
      gsl_rng * r;
};

void free_rgen(struct rgen_t * rgen);

double rand_un();

double gaussian_random(struct rgen_t * rgen, double sigma, double mean);

int generate_random_indices(gsl_vector * v, const int n, struct rgen_t * rgen);

int generate_random_uniform_matrix(gsl_matrix * mat, struct rgen_t * rgen, double min_v);

int generate_random_uniform_vector(gsl_vector * vec, double scale);

void init_rgen(struct rgen_t * rgen, const unsigned long int seed);

// Assumes that the given probabilities are the cumulative sum
int sample_from_cdf(const double * cdf, struct rgen_t * rgen);

double uniform_random(struct rgen_t * rgen);

int generate_random_matrix(gsl_matrix * mat, const double sigma, double mean,struct rgen_t * rgen);

int uniform_random_int_in_range(struct rgen_t * rgen, const int n);

int generate_woodruff_matrix(gsl_matrix * mat, struct rgen_t * rgen);

int generate_osnap_matrix(gsl_matrix * mat, struct rgen_t * rgen);

int generate_aggregate_matrix(gsl_matrix * mat, int b);

/* Matrix and vector functions */
int gsl_matrix_print(gsl_matrix * m);

int gsl_matrix_vector_product(gsl_vector *c, gsl_matrix *A, gsl_vector *b);

// TODO: should each directory have its own utils, as mostly separate utils needed?
//int gsl_matrix_product_A_equal_AB(gsl_matrix *A, gsl_matrix *B, gsl_matrix *work);

int gsl_vector_contains(gsl_vector * a, const int j);

int gsl_vector_print(gsl_vector * v);

int normalize_rows_matrix(gsl_matrix * mat);

int normalize_vector(gsl_vector * vec);

int gsl_outer_product(gsl_matrix *A, gsl_vector *v1, gsl_vector *v2);

int InitPHI(gsl_matrix *phi, const gsl_matrix *oneinds);

int gsl_blas_spddot(const gsl_vector * v1, const gsl_vector * v2_1inds, double *result);
    
int gsl_blas_dgespmv(gsl_matrix * A1ind, const gsl_vector * v1, gsl_vector * v2);

#endif
