#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "algorithm.h"
#include "matrix_algorithms.h"



// TODO: check parameters
const static struct{
      const char *name;
      AlgUpdateFcn update_fcn;
      MatrixUpdateFcn update_mat;
      TraceUpdateFcn update_trace;
} list_matrix_algorithms [] = {
      { "TLSTD", T_LSTD, update_mat_svd, update_trace_replacing},
      { "ATD2nd-TrueA", ATD_2ndorder_trueA, NULL, update_trace_replacing},
      { "ATD2nd", ATD_2ndorder, update_mat_svd, update_trace_replacing},
      { "LSTD", LSTD_lambda, update_mat_sherman, update_trace_replacing},
};
const static int num_totalalgs = (sizeof(list_matrix_algorithms) / sizeof(list_matrix_algorithms[0]));


void init_matrix_alg(struct alg_t * alg, const int numobservations){

      // alg_vars is a void *, initialized here to the correct struct size
      alg->alg_vars = calloc(1, sizeof(struct matrix_alg_vars_t));
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg->alg_vars;

      // TODO: different way to give numfeatures, right now assuming MDP giving observations = features
      int numfeatures = numobservations;
   
      vars->w = gsl_vector_calloc(numfeatures);
      vars->e = gsl_vector_calloc(numfeatures);

      int i;
      // Get the update functions for this algorithm
      for (i = 0; i < num_totalalgs; i++) {
         //printf("name is %s\n", alg->name);
         if (!strcmp(list_matrix_algorithms[i].name, alg->name)) {
            alg->update_fcn = list_matrix_algorithms[i].update_fcn;
            alg->reset_fcn = reset_matrix_alg;
            vars->update_trace = list_matrix_algorithms[i].update_trace;
            vars->update_mat = list_matrix_algorithms[i].update_mat;
            break;
         }
      }
      // All algorithm use a dot product to compute the value function
      alg->get_values = compute_values_matrix;
      
      vars->mvars = NULL;
      vars->mvarsC = NULL;
      vars->work = NULL;
    
      vars->matA = NULL;
      vars->matC = NULL;

      vars->F = 0;
      vars->D = 0;
      vars->I = 1;
      vars->M = 0;
      // TODO: find a better way to set mvar_params
      struct mvar_params_t mvar_params;
      mvar_params.r = 4;
      mvar_params.max_r = 2*mvar_params.r;
      mvar_params.threshold = 0.01;
           
      /*
       * Algorithm specific initializations
       */
      // TODO: currently, ATD2nd would be matched for anything starting with ATD2nd
      // should change to use strncmp
      if (strcmp(alg->name, "TLSTD") == 0) {
         const char * mattype = "low_rank";
         vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "ATD2nd") == 0 ){
         const char * mattype = "low_rank";
         vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "ATD2nd-TrueA") == 0){
          vars->matA = gsl_matrix_calloc(numfeatures, numfeatures);
      }
      else if(strcmp(alg->name, "LSTD") == 0){
          const char * mattype = "full";
          vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
}

void deallocate_matrix_alg(void * alg_vars){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    
      gsl_vector_free(vars->w);
      gsl_vector_free(vars->e);
      
      if (vars->mvars != NULL) {
         deallocate_matrix_vars(vars->mvars);
      }
      if (vars->mvarsC != NULL) {
         deallocate_matrix_vars(vars->mvarsC);
      }
      if (vars->work != NULL) {
         gsl_vector_free(vars->work);
      }
      if (vars->matA != NULL) {
         gsl_matrix_free(vars->matA);
      }
      if (vars->matC != NULL) {
         gsl_matrix_free(vars->matC);
      }
}

void reset_matrix_alg(void * alg_vars){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

      gsl_vector_set_zero(vars->w);
      gsl_vector_set_zero(vars->e);

      vars->t = 0;

      vars->I = 1;
      vars->M = 0;
      vars->F = 0;
      vars->D = 0;
      
      if (vars->mvars != NULL) {
         reset_matrix_vars(vars->mvars);
      }
      if (vars->mvarsC != NULL) {
         reset_matrix_vars(vars->mvarsC);
      }
      if (vars->work != NULL) {
         gsl_vector_set_zero(vars->work);
      }
}


// Assumes that the SVD of A and b was updated outside of this function
int T_LSTD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info) {
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

      //update z vector
      vars->update_trace(vars->e, params, info);
    
      update_bvec(vars->e, info, vars->mvars);
    
      compute_dvec(vars->mvars->dvec, info);
    
      vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);
      // compute weight vector
      compute_weights(vars->w, vars->mvars);
  
      return 0;
}

int ATD_2ndorder_trueA(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    //update z vector
    vars->update_trace(vars->e, params, info);
    
    //compute delta
    double delta = compute_delta(vars->w, info);
    
    gsl_blas_dgemv (CblasNoTrans, delta*params->alpha_t/((double)vars->mvars->t + 1.0), vars->matA, vars->e, 1.0, vars->w);
    
    //add the regularization part
    gsl_blas_daxpy(delta*params->beta_t, vars->e, vars->w);
    
    vars->mvars->t++;
    
    return 0;
}

int ATD_2ndorder(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    if (vars->mvars->t == 0) {
        vars->mvars->threshold = params->threshold;
    }   
    vars->update_trace(vars->e, params, info);
    double delta = compute_delta(vars->w, info);
    //gsl_vector_view xsub = gsl_vector_subvector(info->x_t, 1000, 24);
   
    compute_dvec(vars->mvars->dvec, info);
    
    vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);

    double stepsize = 1.0/((double)vars->mvars->t);

    update_weights(vars->w, vars->e, delta*stepsize, 1.0, vars->mvars, MAT_SVD_INV);
    gsl_blas_daxpy(delta*params->beta_t, vars->e, vars->w);
    
    return 0;
}

// modify: can do both svd or directly maintain A^inv
int LSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    
    if (vars->mvars->t == 0) {
        //gsl_matrix_set_identity(vars->mvars->mat_main);
        gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_main);
        gsl_vector_set_all(&diagA.vector, params->eta_t);
    }

    //update z vector
    vars->update_trace(vars->e, params, info);
    
    //update b vector
    //gsl_blas_daxpy(info->reward, vars->e, vars->mvars->bvec);
    
    update_bvec(vars->e, info, vars->mvars);
    
    compute_dvec(vars->mvars->dvec, info);
    
    vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);
    
    //op_mat_vector_mul(vars->w, vars->mvars->bvec, vars->mvars, MAT_FULL_INV);

    update_weights(vars->w, vars->mvars->bvec, 1, 0, vars->mvars, MAT_FULL);

    return 1;
}


void compute_values_matrix(gsl_vector * values, const gsl_matrix * observations, const void * alg_vars) {
    struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    
    gsl_blas_dgemv (CblasNoTrans, 1.0, observations, vars->w, 0, values);
}
