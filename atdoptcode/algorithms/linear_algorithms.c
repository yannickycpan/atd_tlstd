#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "linear_algorithms.h"

void compute_values_linear(gsl_vector * values, const gsl_matrix * observations, const void * alg_vars);

// TODO: put in correct traces, and decide what to do about trace update fcns
const static struct{
      const char *name;
      AlgUpdateFcn update_fcn;
      TraceUpdateFcn update_trace;
} list_linear_algorithms [] = {
      { "TD", TD_lambda, update_trace_replacing},
      { "ETD", ETD, update_trace_replacing},
      { "TO-TD", TO_TD_lambda, update_trace_trueonline},
      { "TO-ETD", TO_ETD, update_trace_replacing},
      { "TD-Adam", TD_Adam_lambda, update_trace_replacing},
      { "TD-AdaGrad", TD_AdaGrad_lambda, update_trace_replacing},
      { "TD-RMSProp", TD_RMSProp_lambda, update_trace_replacing},
      { "TD-AMSGrad", TD_AMSGrad_lambda, update_trace_replacing},
};
const static int num_totalalgs = (sizeof(list_linear_algorithms) / sizeof(list_linear_algorithms[0]));

void init_linear_alg(struct alg_t * alg, const int numobservations){

      // alg_vars is a void *, initialized here to the correct struct size
      alg->alg_vars = malloc(sizeof(struct linear_alg_vars_t));
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg->alg_vars;

      // TODO: different way to give numfeatures, right now assuming MDP giving observations = features
      int numfeatures = numobservations;
      
      vars->w = gsl_vector_calloc(numfeatures);
      vars->e = gsl_vector_calloc(numfeatures);

      int i;
      // Get the update functions for this algorithm
      for (i = 0; i < num_totalalgs; i++) {
         if (!strcmp(list_linear_algorithms[i].name, alg->name)) {
            alg->update_fcn = list_linear_algorithms[i].update_fcn;
            alg->reset_fcn = reset_linear_alg;
            vars->update_trace = list_linear_algorithms[i].update_trace;
            break;
         }
      }
      // All algorithm use a dot product to compute the value function
      alg->get_values = compute_values_linear;

      vars->eh = NULL;
      vars->h = NULL;
      vars->w_tm1 = NULL;
      vars->work = NULL;
      vars->work1 = NULL;
      vars->mvec = NULL;
      vars->vvec = NULL;
      vars->t = 0;
      vars->VofS = 0; 

      /*
       * Algorithm specific initializations
       */
      if (strcmp(alg->name, "TO-ETD") == 0) {
         vars->work = gsl_vector_alloc(numfeatures);
         vars->I = 1;
      }
      else if(strcmp(alg->name, "TD-Adam") == 0){
         vars->mvec = gsl_vector_alloc(numfeatures);
         vars->vvec = gsl_vector_alloc(numfeatures);
         vars->work = gsl_vector_alloc(numfeatures);
         vars->work1 = gsl_vector_alloc(numfeatures);
      }
      else if(strcmp(alg->name, "TD-AdaGrad") == 0){
        vars->h = gsl_vector_alloc(numfeatures);
        vars->work = gsl_vector_alloc(numfeatures);
        vars->work1 = gsl_vector_alloc(numfeatures);
      }
      else if(strcmp(alg->name, "TD-RMSProp") == 0){
        vars->h = gsl_vector_alloc(numfeatures);
        vars->work = gsl_vector_alloc(numfeatures);
        vars->work1 = gsl_vector_alloc(numfeatures);
      }
      else if(strcmp(alg->name, "TD-AMSGrad") == 0){
          vars->mvec = gsl_vector_alloc(numfeatures);
          vars->vvec = gsl_vector_alloc(numfeatures);
          //h here used as previous vvec
          vars->h = gsl_vector_alloc(numfeatures);
          vars->work = gsl_vector_alloc(numfeatures);
          vars->work1 = gsl_vector_alloc(numfeatures);
      }
}

void deallocate_linear_alg(void * alg_vars){
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
      
      gsl_vector_free(vars->w);
      gsl_vector_free(vars->e);

      if (vars->work != NULL) {
         gsl_vector_free(vars->work);
      }
      if (vars->work1 != NULL) {
         gsl_vector_free(vars->work1);
      }
      if (vars->h != NULL) {
         gsl_vector_free(vars->h);
      }
      if (vars->eh != NULL) {
         gsl_vector_free(vars->eh);
      }
      if (vars->w_tm1 != NULL) {
         gsl_vector_free(vars->w_tm1);
      }
      if (vars->mvec != NULL) {
         gsl_vector_free(vars->mvec);
      }
      if (vars->vvec != NULL) {
         gsl_vector_free(vars->vvec);
      }

      free(vars);
}

void reset_linear_alg(void * alg_vars){
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;

      gsl_vector_set_zero(vars->w);
      gsl_vector_set_zero(vars->e);

      vars->t = 0;

      vars->I = 1;
      vars->M = 0;
      vars->F = 0;
      vars->D = 0;

      if (vars->h != NULL) {
         gsl_vector_set_zero(vars->h);
      }
      if (vars->eh != NULL) {
         gsl_vector_set_zero(vars->eh);
      }
      if (vars->w_tm1 != NULL) {
         gsl_vector_set_zero(vars->w_tm1);
      }
      if (vars->work != NULL) {
         gsl_vector_set_zero(vars->work);
      }
      if (vars->work1 != NULL) {
         gsl_vector_set_zero(vars->work1);
      }
      if (vars->mvec != NULL) {
         gsl_vector_set_zero(vars->mvec);
      }
      if (vars->vvec != NULL) {
         gsl_vector_set_zero(vars->vvec);
      }
}


void compute_values_linear(gsl_vector * values, const gsl_matrix * observations, const void * alg_vars) {
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;

      gsl_blas_dgemv (CblasNoTrans, 1.0, observations, vars->w, 0, values);
      // TODO: avoiding utils for now
      //gsl_matrix_vector_product(values, observations, vars->w);
}

int TD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
      //printf("current reward is %f\n", info->reward);
      // printf("current alpha is %f\n", params->alpha_t);
      //gsl_vector_print(info->x_t);
      // Compute delta
      double delta = compute_delta(vars->w, info);
     //printf("current delta is %f\n", delta); 
     // Update trace
      vars->update_trace(vars->e, params, info);
      // Update weights
      gsl_blas_daxpy (params->alpha_t*delta, vars->e, vars->w);
      
      return 0;
}


/* Store current value in VofS, because need that value for the next step */
int TO_TD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info) {
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;

      double new_v = 0;
      gsl_blas_ddot(vars->w, info->x_tp1, &new_v);  
      double delta = info->reward + info->gamma_tp1*new_v - vars->VofS;
      // printf("the current delta is %f\n", delta);
      // Update trace; ignores vars->update_trace, because must use true-online update
      update_trace_trueonline(vars->e, params, info);

      // Update weights
      double dot = 0;
      gsl_blas_ddot(vars->w, info->x_t, &dot);    
      // TODO: include function in info to compute sparse gsl_blas_daxpy
      gsl_blas_daxpy (delta, vars->e, vars->w);
      gsl_blas_daxpy (params->alpha_t*(vars->VofS - dot), info->x_t, vars->w);

      // Save VofS for next step
      vars->VofS = new_v;
    
      return 0;
}

int ETD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info) {
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
     
    vars->F = info->rho_t*info->gamma_t*(vars->F) + vars->I;
    
    double M = params->lambda_t*vars->I + (1 - params->lambda_t)*(vars->F);
    
    gsl_vector_scale(vars->e, info->gamma_t*info->rho_t*params->lambda_t);
    
    gsl_blas_daxpy(M*info->rho_t, info->x_t, vars->e);

    double delta = compute_delta(vars->w, info);

    gsl_blas_daxpy(params->alpha_t*delta, vars->e, vars->w);

    return 0;
}

int TO_ETD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info) {
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
      
      double delta = compute_delta(vars->w, info);
    
      vars->F = vars->F*info->rho_tm1*info->gamma_t + vars->I;
    
      // update eligibility trace
      double M = params->lambda_t*vars->I + (1-params->lambda_t)*vars->F;
      double phie = 0;
      gsl_blas_ddot (info->x_t, vars->e, &phie);
    
      double S = info->rho_t*params->alpha_t* M * (1 - info->rho_t*info->gamma_t*params->lambda_t*phie);
      gsl_vector_scale(vars->e,info->rho_t*info->gamma_t*params->lambda_t);
      gsl_blas_daxpy (S, info->x_t, vars->e);
    
      /* Rich's approach */
      gsl_vector_memcpy(vars->work, info->x_t);
      gsl_vector_scale(vars->work, -params->alpha_t*M*info->rho_t);
      gsl_vector_add(vars->work, vars->e);
      gsl_vector_scale(vars->work, vars->D);
      gsl_blas_daxpy (delta, vars->e, vars->work);
      gsl_vector_add(vars->w, vars->work);
    
      gsl_blas_ddot (vars->work, info->x_tp1, &(vars->D));
    
      return 0;
}


