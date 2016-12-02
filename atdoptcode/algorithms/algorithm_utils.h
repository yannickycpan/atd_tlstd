#ifndef _ALGORITHM_UTILS_H
#define _ALGORITHM_UTILS_H

#define MAX_AGENT_PRAMS 300
#define MAX_PARAM_STRING_LENGTH 100
#define MIN_SVD_VEC_NORM 0.00001
#define MAX_MATRIX_NAME_LEN 50

// define different matrix format and operation(OP)
// i.e. MAT_FULL_INV means do inverse on a full matrix
// MAT_SVD_INV means do inverse on the SVD of a matrix
// MAT_FULL: directly use original matrix maintained in matrix_vars_t structure without using any operation
typedef int MAT_TYPE_OP;
#define MAT_FULL 0
#define MAT_FULL_INV -1
#define MAT_SVD 1
#define MAT_SVD_INV 2
#define MAT_SVD_TRANS 3

#include "../utils.h"
#include "../rl_interface.h"

struct matrix_vars_t {
    
      // matrix type
      char mat_type[MAX_MATRIX_NAME_LEN];
    
      // Current rank
      int current_r;
      // Desired rank
      int r;
      // Max rank when allowing subspace to grow beyond r
      int max_r;
      
      // Store time step to properly normalize
      int t;
      
      // Threshold for computing inverse
      double threshold;

      gsl_vector * bvec;

      // Matrices for SVD
      // Memory is allocated for max_r, and submatrices of current_r are used
      gsl_vector * sigmavec;
      gsl_matrix * matu;
      gsl_matrix * matv;
      gsl_matrix * matl;
      gsl_matrix * matr;
    
      // full matrix, with size = d^2
      gsl_matrix * mat_main;
      gsl_matrix * delta_main;
      gsl_matrix * work_mat_main;
      gsl_vector * work;
      gsl_vector * work1;

      /*********** PRIVATE VARIABLES BEGIN **********/
      
      // Store work variables to avoid reallocating memory and save on copying values
      // WARNING: any variables labeled work explicitly should never be passed to a function
      // that also takes the mvars variable, since whenever mvars is made available, the work
      // variables consitute variables that can be used however and should not store important
      // variables like arguments to the function
      gsl_vector * mvec;
      gsl_vector * nvec;
      gsl_vector * dvec; 
      gsl_vector * work_d;
      gsl_vector * work_r;
      gsl_vector * work_r_2;
      gsl_matrix * work_mat; 
      gsl_matrix * Kmat; // also plays the role of Lhat
      gsl_matrix * Rhat;
};

struct alg_params_t {
      // Step-size
      double alpha_t;
      // regularizer atds
      double beta_t;
      // Trace parameter
      double lambda_t;
      double lambda_tp1;
      // regularizer lstd
      double eta_t;
      // include threshold
      double threshold;
      int ml;
};

struct mvar_params_t {
      int r;
      int max_r;
      double threshold;
};

typedef void  (*MatrixUpdateFcn)(gsl_vector *, gsl_vector *, const struct transition_info_t *, struct matrix_vars_t *);

struct matrix_vars_t * allocate_matrix_vars(const char *mat_type, int numfeatures,  const struct mvar_params_t * mvar_params);

void deallocate_matrix_vars(struct matrix_vars_t * mvars);

void reset_matrix_vars(struct matrix_vars_t * mvars);

void update_bvec(gsl_vector * z, const struct transition_info_t * info, struct matrix_vars_t * mvars);

void update_mat_svd(gsl_vector * z, gsl_vector *d, const struct transition_info_t * info, struct matrix_vars_t * mvars);

void update_mat_sherman(gsl_vector * z, gsl_vector *d, const struct transition_info_t * info, struct matrix_vars_t * mvars);

void update_mat_normal(gsl_vector * z, gsl_vector *d, const struct transition_info_t * info, struct matrix_vars_t * mvars);

void compute_weights(gsl_vector * w, struct matrix_vars_t * mvars);

double compute_delta(gsl_vector * w, const struct transition_info_t * info);

void compute_dvec(gsl_vector *d, const struct transition_info_t * info);

void update_weights(gsl_vector *w, gsl_vector *v, double alpha, double beta, struct matrix_vars_t * mvars, MAT_TYPE_OP matrix_type_op);

void op_mat_vector_mul(gsl_vector * res, gsl_vector * b, struct matrix_vars_t * mvars, MAT_TYPE_OP matrix_type_op);
//char * params_to_string(char param_string[MAX_PARAM_STRING_LENGTH], const struct alg_params_t * params);

char * params_to_string(char param_string[MAX_PARAM_STRING_LENGTH], const struct alg_params_t * params);

void set_params(struct alg_params_t * dest_params, const struct alg_params_t * src_params);

typedef void  (*TraceUpdateFcn)(gsl_vector *, const struct alg_params_t *, const struct transition_info_t *);


void update_trace_accumulating(gsl_vector * e, const struct alg_params_t * params, const struct transition_info_t * info);

/* Only makes sense for sparse features; assumes sparse x_t */
void update_trace_replacing(gsl_vector * e, const struct alg_params_t * params, const struct transition_info_t * info);

void update_trace_trueonline(gsl_vector * e, const struct alg_params_t * params, const struct transition_info_t * info);

void update_trace_to_gtd(gsl_vector * e, const struct alg_params_t * params, const struct transition_info_t * info);

/* Miscellaneous utils functions */
int gsl_matrix_product_A_equal_AB(gsl_matrix *A, gsl_matrix *B, gsl_matrix *work);

#endif
