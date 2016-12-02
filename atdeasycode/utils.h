#ifndef _UTILS_H
#define _UTILS_H
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix_double.h>


struct rgen_t {
   gsl_rng * r;
};

struct ArraySize {
    int n1;
    int n2;
    int n3;
    int n4;
    int n5;
    int n6;
};


/* Allocation and deallocation */
  
void init_array(double * v, const double val, const int len);

int alloc_vectors(gsl_vector * arr[], const int len1, const int len2);

void zero_vectors(gsl_vector * arr[], const int len1);

int free_vectors(gsl_vector * arr[], const int len1);

double gsl_vector_sum (gsl_vector * arr);


/* Miscellaneous functions */

double min(double a, double b);

double log2(double a);

char* itoa(int val, int base);

int int2Binary(gsl_vector * array, int inv, int n);


/* Spectral functions */

double gsl_get_cond_num(gsl_matrix* A);

double gsl_spectral_radius(gsl_matrix * mat);

int SVD_inv(gsl_matrix * A, int k, double tol);


/* Random number generation */

double rand_un();

double randn (double mu, double sigma);

int generate_random_indices(gsl_vector * v, const int n, struct rgen_t * rgen);

void init_rgen(struct rgen_t * rgen, const unsigned long int seed);

void set_rgen_seed(struct rgen_t * rgen, const unsigned long int seed);

double uniform_random(struct rgen_t * rgen);

double gaussian_random(struct rgen_t * rgen, double sigma, double mean);

int uniform_random_inRange(struct rgen_t * rgen, const int n);

void free_rgen(struct rgen_t * rgen);

int generate_random_vector(gsl_vector * vec, const double sigma, double mean,struct rgen_t * rgen);

int generate_random_uniform_vector(gsl_vector * vec, struct rgen_t * rgen);

int generate_random_matrix(gsl_matrix * mat, const double sigma, const double mean, struct rgen_t * rgen);

int generate_random_uniform_matrix(gsl_matrix * mat, struct rgen_t * rgen, double minV);

int sample_from_dist(double * v, gsl_vector * indexs,struct rgen_t * rgen);

int sample_from_dist_nocumsum(const double * probs, gsl_vector * indexs, struct rgen_t * rgen);


/* Simplifying products */

int gsl_matrix_vector_product(gsl_vector *c, gsl_matrix *A, gsl_vector *b);

int gsl_vector_contains(gsl_vector * a, const int j);

double multiply_vec(const double* v1, const double* v2, const int len);

int gsl_matrix_matrix_rect(gsl_matrix *C, gsl_matrix *A, gsl_matrix *B);

double multiply_gsl_vec(gsl_vector *v1, gsl_vector *v2);

int gsl_outer_product(gsl_matrix *A, gsl_vector *v1, gsl_vector *v2);

/* Normalization functions */

int normalize_rows_matrix(gsl_matrix * mat);

int normalize_rows_matrix_cut(gsl_matrix * mat, const int n);

int normalize_cols_matrix(gsl_matrix * mat);

double normdiff(const gsl_vector* v1, const gsl_vector* v2, const int len);

void divide_all( double *vec, const double divisor, const int len);

void divide_all_mat(const int len1, const int len2, double mat[][len2], const double divisor);

/* Error computing functions */

double compute_rmse(const gsl_vector *p, const gsl_vector *t);

double compute_pme(const gsl_vector *p, const gsl_vector *t);


/* Bookkeeping functions */

int gsl_vector_print(gsl_vector * v);

int gsl_matrix_print(gsl_matrix * m);

int FiveDto1DIndex(int a, int b, int c, int d, int e, const struct ArraySize* sizes);

int SixDto1DIndex(int a, int b, int c, int d, int e, int h, const struct ArraySize* sizes);

int compute_mean_variance_of_vector(const gsl_vector * vec, double * mean, double * var);

int write_matrix_to_file(const int n, const int m, double mat[][m], char* name);

int write_matrix_to_file_with_names(const int n, const int m, double mat[][m],  const int nNames, char* names[], char* name);

int write_vector_to_file(const int n, double* vec, char* name);

int print_matrix(const int n, const int m, double mat[][m]);

/* Sparse matrices computation functions */

int gsl_spvector_scale(gsl_vector * v1, const gsl_vector * v1_1inds, double scalor);

int gsl_spvector_add(gsl_vector * v1, const gsl_vector * v2, const gsl_vector * v2_1inds);

int gsl_spvector_sub(gsl_vector * v1, const gsl_vector * v2, const gsl_vector * v2_1inds);

gsl_vector * gsl_spvector_get_nz2vecs(const gsl_vector * v1_1inds, const gsl_vector * v2_1inds);

int gsl_blas_spdaxpy(const double alpha, const gsl_vector *x, const gsl_vector *xnz, gsl_vector* y);

int gsl_spvector_mul(gsl_vector * v1, const gsl_vector * v2, const gsl_vector * v2_1inds);

int gsl_blas_spddot(const gsl_vector * v1, const gsl_vector * v2, const gsl_vector * v2_1inds, double *result);

int gsl_blas_dgemspv(const gsl_matrix * A, const gsl_vector * v1, const gsl_vector * v1_1inds, gsl_vector * v2);

int gsl_blas_dgespmv(const gsl_matrix * A, const gsl_matrix * A1ind, const gsl_vector * v1, gsl_vector * v2);

int gsl_blas_dgespvm(const gsl_matrix * A, const gsl_vector * v1, const gsl_vector * v1_1inds, gsl_vector * v2);

int gsl_blas_dgespmm(const gsl_matrix * A, const gsl_matrix * A1ind, const gsl_matrix * B, gsl_matrix * C);

int gsl_outer_product_sp(gsl_matrix *A, gsl_vector *v1, const gsl_vector * v1_1inds, gsl_vector *v2);

int InitPHI(gsl_matrix *phi, const gsl_matrix *oneinds);

/* QR decomposition */

int gsl_linalg_QR_Qmat (const gsl_matrix * QR, const gsl_vector * tau, gsl_matrix * A);

int gsl_linalg_QR_unpackR (const gsl_matrix * QR, const gsl_vector * tau, const int k, gsl_matrix * R);

#endif
