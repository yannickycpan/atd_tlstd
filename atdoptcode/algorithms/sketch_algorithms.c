#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "algorithm.h"
//#include "matrix_algorithms.h"
#include "sketch_algorithms.h"



// TODO: check parameters
const static struct{
      const char *name;
      AlgUpdateFcn update_fcn;
      MatrixUpdateFcn update_mat;
      TraceUpdateFcn update_trace;
} list_sketch_algorithms [] = {
            { "TEST", TEST_lambda, update_mat_normal, update_trace_accumulating},
};
const static int num_totalalgs = (sizeof(list_sketch_algorithms) / sizeof(list_sketch_algorithms[0]));


void init_sketch_alg(struct alg_t * alg, const int numobservations){
//      printf("init_sketch is called.\n");
      // alg_vars is a void *, initialized here to the correct struct size
      alg->alg_vars = calloc(1, sizeof(struct sketch_alg_vars_t));
      struct sketch_alg_vars_t * vars = (struct sketch_alg_vars_t *) alg->alg_vars;

      // TODO: different way to give numfeatures, right now assuming MDP giving observations = features
      int numfeatures = numobservations;

      vars->w = gsl_vector_calloc(numfeatures);
      vars->e = gsl_vector_calloc(numfeatures);

      int i;
      // Get the update functions for this algorithm
      for (i = 0; i < num_totalalgs; i++) {
            if (!strcmp(list_sketch_algorithms[i].name, alg->name)) {
                  alg->update_fcn = list_sketch_algorithms[i].update_fcn;
                  alg->reset_fcn = reset_sketch_alg;
                  vars->update_trace = list_sketch_algorithms[i].update_trace;
                  vars->update_mat = list_sketch_algorithms[i].update_mat;
                  break;
            }
      }
      // All algorithm use a dot product to compute the value function
      alg->get_values = compute_values_matrix;

      vars->mvars = NULL;
      //      vars->mvarsC = NULL;
      vars->work = NULL;

//      vars->matA = NULL;
      //      vars->matC = NULL;

      // TODO: find a better way to set mvar_params
      struct mvar_params_t mvar_params;
      mvar_params.r = numfeatures;
      mvar_params.max_r = 2*mvar_params.r;
      mvar_params.threshold = 0.01;

      /*
       * Algorithm specific initializations
       */
      // TODO: currently, ATD2nd would be matched for anything starting with ATD2nd
      // should change to use strncmp
      if(strcmp(alg->name, "TEST") == 0){
            const char * mattype = "full";
            vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
}
//
void deallocate_sketch_alg(void * alg_vars){
      struct sketch_alg_vars_t * vars = (struct sketch_alg_vars_t *) alg_vars;

      gsl_vector_free(vars->w);
      gsl_vector_free(vars->e);

      if (vars->mvars != NULL) {
            deallocate_matrix_vars(vars->mvars);
      }
      //      if (vars->mvarsC != NULL) {
      //         deallocate_matrix_vars(vars->mvarsC);
      //      }
      if (vars->work != NULL) {
            gsl_vector_free(vars->work);
      }
//      if (vars->matA != NULL) {
//            gsl_matrix_free(vars->matA);
//      }
      //      if (vars->matC != NULL) {
      //         gsl_matrix_free(vars->matC);
      //      }
}
//
void reset_sketch_alg(void * alg_vars){
      struct sketch_alg_vars_t * vars = (struct sketch_alg_vars_t *) alg_vars;

      gsl_vector_set_zero(vars->w);
      gsl_vector_set_zero(vars->e);

      if (vars->mvars != NULL) {
            reset_matrix_vars(vars->mvars);
      }
}
//
//
// directly maintain A^inv
int TEST_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
//      printf("TEST_lambda is called.\n");
      struct sketch_alg_vars_t * vars = (struct sketch_alg_vars_t *) alg_vars;

      if (vars->mvars->t == 0) {
            //gsl_sketch_set_identity(vars->mvars->mat_main);
            //gsl_sketch_scale(vars->mvars->mat_main, params->alpha_t);
      }

      //update z vector
      vars->update_trace(vars->e, params, info);

      //update b vector
      //gsl_blas_daxpy(info->reward, vars->e, vars->mvars->bvec);

      update_bvec(vars->e, info, vars->mvars);

      compute_dvec(vars->mvars->dvec, info);
//      VAR(vars->e->size);
//      VAR(vars->mvars->dvec->size);
      vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);
      op_mat_vector_mul(vars->w, vars->mvars->bvec, vars->mvars, MAT_FULL_INV);
//      printf("TEST_lambda is done.\n");
      return 1;
}
