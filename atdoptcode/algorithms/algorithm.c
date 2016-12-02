#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "algorithm.h"


void init_alg(struct alg_t * alg, AlgInitFcn init_fcn, const int numobservations, const char * name){

      // All algorithms have named saved, so do in outer function
      memset(alg->name, 0, MAX_AGENT_NAME_LEN);
      strcpy(alg->name, name);
      
      init_fcn(alg, numobservations);
}

void deallocate_alg(struct alg_t * alg, AlgDeallocateFcn deallocate_fcn){
      deallocate_fcn(alg->alg_vars);
}

void reset_alg(struct alg_t * alg){
      alg->reset_fcn(alg->alg_vars);
}

void update_alg(struct alg_t * alg, struct transition_info_t * tinfo) {
      alg->update_fcn(alg->alg_vars, &(alg->params), tinfo);
}

