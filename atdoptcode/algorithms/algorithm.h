#ifndef _ALGORITHM_H
#define _ALGORITHM_H

#include "algorithm_utils.h"

#define MAX_AGENT_NAME_LEN 100
// typedef the function pointers for updating to make it more readable
typedef int  (*AlgUpdateFcn)(void *, const struct alg_params_t *, const struct transition_info_t *);
typedef void  (*AlgResetFcn)(void *);
// Computes the value for the given states, putting the result into gsl_vector 
// matrix of observations, with each row corresponding to a (sampled) state
// the void * is the algorithm variables
typedef void  (*ValueFcn)(gsl_vector *, const gsl_matrix *, const void *); 

      
struct alg_t {
      char name[MAX_AGENT_NAME_LEN];
      
      // Algorithm parameters; not using a pointer since params is small and avoid dynamic allocation
      // TODO: consider removing this from the algorithm, because it is passed to update_fcn
      // and so does not need to be saved here
      struct alg_params_t params;

      AlgResetFcn reset_fcn;
      AlgUpdateFcn update_fcn;
      ValueFcn get_values;
     
      // sparse computing indicator
      int sparse; 
      // Generic algorithm variables
      void * alg_vars;
};

// Allocation and deallocation functions provided by the algorithms
typedef void  (*AlgInitFcn)(struct alg_t *, const int);
typedef void  (*AlgDeallocateFcn)(void *);


// TODO: consider passing more general initialization struct, rather than just numfeatures, for other useful values for initialization
void init_alg(struct alg_t * alg, AlgInitFcn init_fcn, const int numfeatures, const char * name);

void deallocate_alg(struct alg_t * alg, AlgDeallocateFcn deallocate_fcn);

void reset_alg(struct alg_t * alg);

void update_alg(struct alg_t * alg, struct transition_info_t * tinfo);


#endif
