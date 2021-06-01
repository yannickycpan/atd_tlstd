#ifndef _MATRIX_ALGORITHMS_H
#define _MATRIX_ALGORITHMS_H

#include "algorithm_utils.h"

// typedef the function pointers for updating to make it more readable
//typedef void  (*MatrixUpdateFcn)(gsl_vector *, gsl_vector *, const struct transition_info_t *, struct matrix_vars_t *);


// Algorithms that use A matrix could use different A matrices or share the same A
// For time reasons, we should share any A matrices instead of recomputing for each algorithm
// TODO: for now, for elegance, not implementing this time saving measure

struct matrix_alg_vars_t {
      
      // Primary weights
      gsl_vector * w;
      // Eligibility trace
      gsl_vector * e;

      // Function to update eligibility trace, e.g., accumulating or replacing
      TraceUpdateFcn update_trace;

      // Function to update key matrix
      MatrixUpdateFcn update_mat;
    
    
      // For some algorithms require additional working space, requires to init separately
      gsl_vector *work;
    
      // For true A and C
      gsl_matrix *matA;
      gsl_matrix *trueA;
      gsl_matrix *matC;

      // For algorithms that use an approximation to A
      struct matrix_vars_t * mvars;
    
      // For algorithms that use an approximation to C
      struct matrix_vars_t * mvarsC;
     
      int t;
      double F, D, I, M;

};
//Erfan: I moved the following declaration from .c to here. Is that fine?
void compute_values_matrix(gsl_vector * values, const gsl_matrix * observations, const void * alg_vars);

// Assumes name already in alg
void init_matrix_alg(struct alg_t * alg, const int numobservations);

void deallocate_matrix_alg(void * alg_vars);

void reset_matrix_alg(void * alg_vars);


int T_LSTD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int LSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_2ndorder(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_2ndorder_trueA(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_2ndorder_fullA(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

#endif
