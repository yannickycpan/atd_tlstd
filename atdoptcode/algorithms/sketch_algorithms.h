#ifndef _SKETCH_ALGORITHMS_H
#define _SKETCH_ALGORITHMS_H

#include "algorithm_utils.h"
#include "matrix_algorithms.h"


// typedef the function pointers for updating to make it more readable
//typedef void  (*MatrixUpdateFcn)(gsl_vector *, gsl_vector *, const struct transition_info_t *, struct matrix_vars_t *);


// Algorithms that use A matrix could use different A matrices or share the same A
// For time reasons, we should share any A matrices instead of recomputing for each algorithm
// TODO: for now, for elegance, not implementing this time saving measure

struct sketch_alg_vars_t {
      
      // Primary weights
      gsl_vector * w;
      // Eligibility trace aka z
      gsl_vector * e;

      // Function to update eligibility trace, e.g., accumulating or replacing
      TraceUpdateFcn update_trace;

      // Function to update key matrix
      MatrixUpdateFcn update_mat;
    
    
      // For some algorithms require additional working space, requires to init separately
      gsl_vector *work;
    


      //Sketching Struct:
      int left_dim;
      int right_dim;
      gsl_vector * w_sketch;//The smaller sketched w
      gsl_vector * e_sketch;//The smaller sketched e eligibility trace
      gsl_matrix * left_sketch;//The projection matrix from left
      gsl_matrix * right_sketch;//The projection matrix from right

      // For algorithms that use an approximation to A
      struct matrix_vars_t * mvars;//This is sketched A!!
    
};

// Assumes name already in alg
void init_sketch_alg(struct alg_t * alg, const int numobservations);
//
void deallocate_sketch_alg(void * alg_vars);
//
void reset_sketch_alg(void * alg_vars);
//
int TEST_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

#endif
