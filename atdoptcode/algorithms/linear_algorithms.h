#ifndef _LINEAR_ALGORITHMS_H
#define _LINEAR_ALGORITHMS_H

//#include "utils.h"
#include "algorithm.h"
#include "algorithm_utils.h"

struct linear_alg_vars_t {
      
      // Primary weights
      gsl_vector * w;
      // Eligibility trace
      gsl_vector * e;

      // Function to update eligibility trace, e.g., accumulating or replacing
      TraceUpdateFcn update_trace;
    
      // For true-online methods, save previous value at state s to save computations
      double VofS;
    
      // For true-online ETD
      double F, D, I, M;
    
      // For GTD and TO-GTD
      gsl_vector *h, *eh, *w_tm1;

      // For Adam TD
      gsl_vector *vvec, *mvec;
    
      // For some algorithms require additional working space, requires to init separately
      gsl_vector *work;
      gsl_vector *work1;

      //sometimes need time index t
      int t;
};


// Assumes name already in alg
void init_linear_alg(struct alg_t * alg, const int numobservations);

// void * will be cast to a linear_alg_vars
void deallocate_linear_alg(void * alg_vars);

void reset_linear_alg(void * alg_vars);

int TD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int TD_Adam_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int TD_AdaGrad_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int TD_AMSGrad_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int TD_RMSProp_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int TO_TD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ETD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int TO_ETD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);


#endif
