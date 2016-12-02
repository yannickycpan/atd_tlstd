#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "boyan_mdp.h"
#include "utils.h"

/*
 * Default model_mdp
 */
const struct model_mdp_opts_t default_model_mdp_opts = {.branch = 1, .gamma = 1, .numactions = 2, .numstates = 12,
            .pi_prob = 0.5, .mu_prob = 0.5, .offpolicy = 0,
            .observation_type = "tabular", .reward_type = "random_uniform", .seed = 0};


/***** Private functions used below, in alphabetical order *****/
void compute_Amat(double lambda, gsl_matrix *Amat, const struct model_mdp_t * model_mdp);
void compute_bvec(double lambda, gsl_vector * b, const struct model_mdp_t * model_mdp);
void compute_Cmat(gsl_matrix * matC, const struct model_mdp_t * model_mdp);
void compute_expected_reward(gsl_vector * expected_reward, const struct model_mdp_t * model_mdp);
void compute_Pmu(gsl_matrix * Pmu, const struct model_mdp_t * model_mdp);
void compute_ppi_gamma(gsl_matrix * Ppigamma, const struct model_mdp_t * model_mdp);
void create_rewards(struct model_mdp_t * model_mdp, const struct model_mdp_opts_t * opts);
int generate_cumulative_sum_P_SA(double probs[MAX_ACTIONS][MAX_STATES][MAX_STATES],const struct model_mdp_t * model_mdp);
int generate_cumulative_sum_policy(double probs[MAX_STATES][MAX_ACTIONS],const struct model_mdp_t * model_mdp);
void get_dmu(gsl_vector * dmu, const struct model_mdp_t * model_mdp);
int generate_P_matrix(gsl_matrix * P [MAX_ACTIONS], const int na, const int branch, struct rgen_t *rgen);
void get_features(struct model_mdp_t * model_mdp, const struct model_mdp_opts_t * opts);
int get_stationary_dist(gsl_vector *dmu, const gsl_matrix * Pmu);
int get_vstar(gsl_vector * vstar, const struct model_mdp_t * model_mdp);
int make_model_mdp(struct model_mdp_t * model_mdp, const struct model_mdp_opts_t * opts);
void make_policy(gsl_matrix * pi, double main_action_prob, const struct model_mdp_t * model_mdp);

// given a state, compute its immediate next state
void nextstate(gsl_vector *state);

void nextstate(gsl_vector *state){
      for (int i = 0; i<state->size-1; i++) {
            if (gsl_vector_get(state,i) > 0) {
                  gsl_vector_set(state, i, gsl_vector_get(state,i) - 0.25);
                  gsl_vector_set(state, i+1, gsl_vector_get(state,i+1) + 0.25);
                  return;
            }
      }
}
/***** End Private functions used below ************************/

void deallocate_model_mdp_t(struct model_mdp_t *model_mdp){

      gsl_matrix_free(model_mdp->true_observations);
      gsl_matrix_free(model_mdp->pi);
      gsl_matrix_free(model_mdp->mu);
      for (int a = 0; a < model_mdp->numactions; a++) {
            gsl_matrix_free(model_mdp->gammas[a]);
            gsl_matrix_free(model_mdp->rewards[a]);
            gsl_matrix_free(model_mdp->P[a]);
      }

      free_rgen(model_mdp->rgen);
      free(model_mdp->rgen);

      free(model_mdp);
}


struct model_mdp_t * get_model_mdp(const char * filename) {
      struct model_mdp_t * model_mdp = malloc(sizeof(struct model_mdp_t));

      if (filename == NULL) {
            make_model_mdp(model_mdp, &default_model_mdp_opts);
      } else {
            // Read in model_opts from file
            struct model_mdp_opts_t *opts = malloc(sizeof(struct model_mdp_opts_t));
            FILE * file= fopen(filename, "rb");
            if (file != NULL) {
                  fread(opts, sizeof(struct model_mdp_opts_t), 1, file);
                  fclose(file);
            }
            make_model_mdp(model_mdp, opts);
      }

      return model_mdp;
}

void reset_model_mdp(gsl_vector * x_0, struct model_mdp_t * model_mdp) {
      // Jump to a random start state
      model_mdp->s_t = 0;

      gsl_vector_view x = gsl_matrix_row(model_mdp->true_observations, model_mdp->s_t);
      gsl_vector_memcpy(x_0, &x.vector);      
}

void env_step_model_mdp(gsl_vector * x_tp1, double * reward, double * gamma_tp1, struct model_mdp_t *model_mdp) {
      int s_t = model_mdp->s_t;
      int a_t = sample_from_cdf(model_mdp->A_cdf[s_t], model_mdp->rgen);
      int s_tp1 = sample_from_cdf(model_mdp->SA_cdf[a_t][s_t], model_mdp->rgen);
      // Set the features according to this transition

      *gamma_tp1 = gsl_matrix_get(model_mdp->gammas[a_t], s_t, s_tp1);

      *reward = gsl_matrix_get(model_mdp->rewards[a_t], s_t, s_tp1);

      gsl_vector_view x = gsl_matrix_row(model_mdp->true_observations, s_tp1);
      gsl_vector_memcpy(x_tp1, &x.vector);

      // For next step
      model_mdp->s_t = s_tp1;
}

void get_model_Amat(double lambda, gsl_matrix * Amat, struct model_mdp_t *model_mdp){
      compute_Amat(lambda, Amat, model_mdp);
}

void get_model_Cmat(gsl_matrix * matC, struct model_mdp_t *model_mdp){
      compute_Cmat(matC, model_mdp);
}

/************************ Private functions to create MODEL_MDP *********************/

int make_model_mdp(struct model_mdp_t * model_mdp, const struct model_mdp_opts_t * opts) {

      int a;

      // Initialize variables that do not yet need numobservations info
      model_mdp->numstates = opts->numstates;
      model_mdp->numactions = opts->numactions;
      model_mdp->mu = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numactions);
      model_mdp->pi = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numactions);
      for (a = 0; a < model_mdp->numactions; a++) {
            model_mdp->gammas[a] = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numstates);
            model_mdp->rewards[a] = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numstates);
            model_mdp->P[a] = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numstates);
      }
      model_mdp->rgen = malloc(sizeof(struct rgen_t));      
      init_rgen(model_mdp->rgen, opts->seed);

      /* Generate constant random gamma for now */
      for (a = 0; a < opts->numactions; a++) {
            gsl_matrix_set_all(model_mdp->gammas[a], 1.0);
            gsl_matrix_set(model_mdp->gammas[a], model_mdp->numstates - 1, 0, 0);
            if (a == 1) {
                  gsl_matrix_set(model_mdp->gammas[a], model_mdp->numstates - 2, 0, 0);
            }
      }
      //assume action 1 goes one step; action 2 goes two step
      /* Create rewards based on type */
      create_rewards(model_mdp, opts);

      // Default action probability is (1-mainprob)/(remaining actions)
      make_policy(model_mdp->pi, opts->pi_prob, model_mdp);
      if (opts->offpolicy == 0) {
            gsl_matrix_memcpy (model_mdp->mu, model_mdp->pi);
      } else {
            make_policy(model_mdp->mu, opts->mu_prob, model_mdp);
      }

      /* generate transition matrix */
      generate_P_matrix(model_mdp->P, model_mdp->numactions, opts->branch,model_mdp->rgen);

      generate_cumulative_sum_policy(model_mdp->A_cdf, model_mdp);
      generate_cumulative_sum_P_SA(model_mdp->SA_cdf, model_mdp);

      /* get observations matrix */
      get_features(model_mdp, opts);
      model_mdp->numobservations = model_mdp->true_observations->size2;

      // Get vstar and dmu for error computations
      model_mdp->vstar = gsl_vector_calloc(model_mdp->numstates);
      model_mdp->dmu = gsl_vector_calloc(model_mdp->numstates);
      get_vstar(model_mdp->vstar, model_mdp);

      printf("the true value vec is ---------------\n");
      gsl_vector_print(model_mdp->vstar);
      get_dmu(model_mdp->dmu, model_mdp);
      /*
      double wt[4] = {-24, -16, -8, 0};
      gsl_vector_view wtvec = gsl_vector_view_array(wt, 4);


      gsl_blas_dgemv (CblasNoTrans, 1, model_mdp->true_observations, &wtvec.vector, 0, model_mdp->vstar);

      printf("the values computed by linear solution is ---------------\n");
      gsl_vector_print(model_mdp->vstar);

      get_dmu(model_mdp->dmu, model_mdp);

      printf("the mu vec is ---------------\n");
      gsl_vector_print(model_mdp->dmu);

      gsl_matrix *mata = gsl_matrix_alloc(4, 4);
      compute_Amat(1, mata, model_mdp);
      printf("the A matrix is ---------------\n");
      gsl_matrix_print(mata);

      int s;
      gsl_permutation * p = gsl_permutation_alloc (4);
      gsl_linalg_LU_decomp (mata, p, &s);
      // Invert the matrix m
      gsl_matrix *work2 = gsl_matrix_calloc(4, 4);
      gsl_linalg_LU_invert (mata, p, work2);

      gsl_vector *bvec = gsl_vector_calloc(4);
      compute_bvec(1, bvec, model_mdp);

      gsl_vector *wgt = gsl_vector_calloc(4);
      gsl_blas_dgemv(CblasNoTrans, 1.0, work2, bvec, 0, wgt);

      printf("the exact linear solution by using true A and b is ----------\n");
      gsl_vector_print(wgt);
    

      gsl_permutation_free(p);
      gsl_vector_free(bvec);
      gsl_vector_free(wgt);
      gsl_matrix_free(work2);
      gsl_matrix_free(mata);
      */
    
      return 0;
}

void make_policy(gsl_matrix * pi, double main_action_prob, const struct model_mdp_t * model_mdp) {
      int s, main_action;
      gsl_matrix_set_all(pi,(1.0-main_action_prob)/(model_mdp->numactions-1));
      for(s = 0;s < model_mdp->numstates; s++)
      {
            main_action = uniform_random_int_in_range(model_mdp->rgen, model_mdp->numactions);
            gsl_matrix_set(pi, s, main_action, main_action_prob);
      }
      normalize_rows_matrix(pi);
}

void create_rewards(struct model_mdp_t * model_mdp, const struct model_mdp_opts_t * opts) {
      /* Currently, reward is only about s_t -> s_tp1, regardless of action */
      int a;
      for (a = 0; a < opts->numactions; a++){
            gsl_matrix_set_all (model_mdp->rewards[a], 0);
            int i = 0;
            for (i = 0; i < model_mdp->rewards[a]->size1 - a - 1; i++) {
                  gsl_matrix_set(model_mdp->rewards[a], i, i + 1 + a, -3);
            }
            if (a == 1) {
                  gsl_matrix_set(model_mdp->rewards[a], model_mdp->numstates - 2, 0, -3);
            }
            gsl_matrix_set(model_mdp->rewards[a], model_mdp->numstates - 1, 0, -2);
            //printf("---------------reward--------------\n");
            //gsl_matrix_print(model_mdp->rewards[a]);
      }
}

int generate_P_matrix(gsl_matrix * P [MAX_ACTIONS], const int na, const int branch, struct rgen_t *rgen){
      /* loop over actions */
      for (int a=0; a<na; a++) {
            gsl_matrix_set_all(P[a],0);
            for (int i = 0; i < P[a]->size1-1-a; i++) {
                  gsl_matrix_set(P[a], i, i+a+1, 1.0);
            }
            if (a == 1) {
                  gsl_matrix_set(P[a], P[a]->size1-2, 0, 1.0);
            }
            //no matter what action it takes at 11th state, must go to the initial state
            gsl_matrix_set(P[a], P[a]->size1-1, 0, 1.0);
            //printf("--------------prob mat---------------\n");
            //gsl_matrix_print(P[a]);
      }
      return 0;
}

/* Functions used to make sampling faster */
int generate_cumulative_sum_policy(double probs[MAX_STATES][MAX_ACTIONS],const struct model_mdp_t * model_mdp)
{
      for (int i=0;i<model_mdp->mu->size1;i++)
      {
            double p= gsl_matrix_get(model_mdp->mu,i,0);
            probs[i][0] = p;
            for(int j=1;j<model_mdp->mu->size2;j++){
                  p += gsl_matrix_get(model_mdp->mu,i,j);
                  probs[i][j] = p;
            }
      }
      return 0;
}

int generate_cumulative_sum_P_SA(double probs[MAX_ACTIONS][MAX_STATES][MAX_STATES],const struct model_mdp_t * model_mdp)
{
      for(int a=0;a<model_mdp->numactions;a++)
      {
            for (int i=0;i<model_mdp->P[a]->size1;i++)
            {
                  double p= gsl_matrix_get(model_mdp->P[a],i,0);
                  probs[a][i][0] = p;
                  for(int j=1;j<model_mdp->P[a]->size2;j++){
                        p += gsl_matrix_get(model_mdp->P[a],i,j);
                        probs[a][i][j] = p;
                  }
            }
      }
      return 0;
}

void get_features(struct model_mdp_t * model_mdp, const struct model_mdp_opts_t * opts) {
    
      int num_features = model_mdp->numstates / 4 + 1;

      model_mdp->true_observations = gsl_matrix_alloc(model_mdp->numstates, num_features);

      gsl_vector *curstate = gsl_vector_calloc(model_mdp->true_observations->size2);
      gsl_vector_set(curstate, 0, 1);
      gsl_vector_view rowPhi = gsl_matrix_row(model_mdp->true_observations, 0);
      gsl_vector_memcpy(&rowPhi.vector, curstate);
      for (int i = 1; i < model_mdp->true_observations->size1; i++) {
            nextstate(curstate);
            rowPhi = gsl_matrix_row(model_mdp->true_observations, i);
            gsl_vector_memcpy(&rowPhi.vector, curstate);
      }
      gsl_vector_free(curstate);
}


/************************ Private functions to create MODEL_MDP *********************/


/*************** Private utility functions ***************/

int get_vstar(gsl_vector * vstar, const struct model_mdp_t * model_mdp) {
      // Compute I - Ppigamma
      gsl_matrix * Ppigamma = gsl_matrix_calloc(model_mdp->numstates,model_mdp->numstates);
      compute_ppi_gamma(Ppigamma, model_mdp);
      gsl_matrix * eye = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numstates);
      gsl_matrix_set_identity(eye);
      gsl_matrix_sub(eye,Ppigamma);
      // Compute expected reward
      gsl_vector * expected_reward = gsl_vector_calloc(model_mdp->numstates);
      compute_expected_reward(expected_reward, model_mdp);
      //gsl_vector_print(expected_reward);
      // Compute vstar as (I - Ppigammma)^{-1} Rbar
      int s;
      gsl_matrix * M = gsl_matrix_alloc (model_mdp->numstates, model_mdp->numstates);
      gsl_permutation * perm = gsl_permutation_alloc (model_mdp->numstates);

      /* Make LU decomposition of matrix eye */
      gsl_linalg_LU_decomp (eye, perm, &s);
      /* Invert the matrix eye put result in M */
      gsl_linalg_LU_invert (eye, perm, M);
      gsl_matrix_vector_product(vstar, M, expected_reward);
      gsl_matrix_free(Ppigamma);
      gsl_matrix_free(eye);
      gsl_matrix_free(M);
      gsl_permutation_free(perm);

      return 1;
}

void compute_ppi_gamma(gsl_matrix * Ppigamma, const struct model_mdp_t * model_mdp) {
      gsl_matrix_set_zero(Ppigamma);

      double sum = 0.0;
      int i, j, a;
      for (i = 0; i < model_mdp->numstates; i++) {
            for (j = 0; j < model_mdp->numstates; j++){
                  sum = 0.0;
                  for (a = 0; a < model_mdp->numactions; a++)
                        sum = sum + gsl_matrix_get(model_mdp->P[a], i, j)*gsl_matrix_get(model_mdp->pi, i, a)*gsl_matrix_get(model_mdp->gammas[a], i, j);
                  gsl_matrix_set(Ppigamma,i,j,sum);
            }
      }
}

void compute_expected_reward(gsl_vector * expected_reward, const struct model_mdp_t * model_mdp){
      /* compute expected rewards */
      gsl_vector_set_zero(expected_reward);
      double sum = 0.0;
      int i, j, a;
      for (i = 0; i < model_mdp->numstates; i++) {
            sum = 0.0;
            for (j = 0; j < model_mdp->numstates; j++){
                  for (a = 0; a < model_mdp->numactions; a++) {
                        sum = sum + gsl_matrix_get(model_mdp->P[a], i, j)*gsl_matrix_get(model_mdp->pi, i, a)*gsl_matrix_get(model_mdp->rewards[a], i, j);
                  }
            }
            gsl_vector_set(expected_reward,i,sum);
      }
}

void get_dmu(gsl_vector * dmu, const struct model_mdp_t * model_mdp) {
      gsl_matrix * Pmu = gsl_matrix_calloc(model_mdp->numstates,model_mdp->numstates);
      compute_Pmu(Pmu, model_mdp);
      get_stationary_dist(dmu, Pmu);
      free(Pmu);
}

void compute_Pmu(gsl_matrix * Pmu, const struct model_mdp_t * model_mdp) {
      gsl_matrix_set_zero(Pmu);

      double sum = 0.0;
      int i, j, a;
      for (i = 0; i < model_mdp->numstates; i++) {
            for (j = 0; j < model_mdp->numstates; j++){
                  sum = 0.0;
                  for (a = 0; a < model_mdp->numactions; a++)
                        sum = sum + gsl_matrix_get(model_mdp->P[a], i, j)*gsl_matrix_get(model_mdp->mu, i, a);
                  gsl_matrix_set(Pmu,i,j,sum);
            }
      }
}

int get_stationary_dist(gsl_vector *dmu, const gsl_matrix * Pmu)
{
      gsl_vector_set_zero(dmu);
      gsl_vector_complex *eval = gsl_vector_complex_alloc (Pmu->size1);
      gsl_matrix_complex *evec = gsl_matrix_complex_alloc (Pmu->size1, Pmu->size1);

      gsl_matrix * A = gsl_matrix_alloc (Pmu->size1, Pmu->size2);
      gsl_matrix_transpose_memcpy(A, Pmu);

      gsl_eigen_nonsymmv_workspace * w =  gsl_eigen_nonsymmv_alloc (Pmu->size1);
      gsl_eigen_nonsymmv (A, eval, evec, w);
      gsl_eigen_nonsymmv_free (w);
      gsl_eigen_nonsymmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_DESC);

      /* get first column corresponding to the eigen vector for the largest eigen value */
      gsl_vector_complex_view evec_c0 = gsl_matrix_complex_column (evec, 0);
      gsl_vector_view evec_0 = gsl_vector_complex_real (&evec_c0.vector);
      gsl_vector_memcpy (dmu, &evec_0.vector);
      normalize_vector(dmu);

      gsl_vector_complex_free (eval);
      gsl_matrix_complex_free (evec);
      gsl_matrix_free (A);

      return 0;  
}


void compute_Amat(double lambda, gsl_matrix * Amat, const struct model_mdp_t * model_mdp){

      gsl_matrix *Phi = model_mdp->true_observations;
      gsl_vector *dmu = model_mdp->dmu;

      gsl_matrix *Pmat = gsl_matrix_calloc(model_mdp->numstates, model_mdp->numstates);
      compute_ppi_gamma(Pmat, model_mdp);
      // Pmat is actually gamma * P

      gsl_matrix *Dmat = gsl_matrix_calloc(dmu->size, dmu->size);
      gsl_matrix *work = gsl_matrix_calloc(Phi->size2, dmu->size);
      gsl_vector_view D_diag = gsl_matrix_diagonal (Dmat);
      gsl_vector_memcpy(&D_diag.vector, dmu);
      gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, Phi, Dmat, 0, work);
      if (lambda == 1) {
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, work, Phi, 0, Amat);
      }
      else if (lambda == 0){
            gsl_matrix * eye = gsl_matrix_alloc(model_mdp->numstates, model_mdp->numstates);
            gsl_matrix_set_identity(eye);
            gsl_matrix_sub(eye, Pmat);
            gsl_matrix *work3 = gsl_matrix_alloc(Phi->size2, Phi->size1);
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, work, eye, 0, work3);
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, work3, Phi, 0, Amat);
            gsl_matrix_free(work3);
            gsl_matrix_free(eye);
      }
      else{
            gsl_matrix_scale(Pmat, lambda);
            gsl_matrix * eye = gsl_matrix_alloc(model_mdp->numstates, model_mdp->numstates);
            gsl_matrix_set_identity(eye);
            gsl_matrix_sub(eye, Pmat);
            //now eye is I - lambda gamma P
            // Make LU decomposition of matrix m
            int s = 0;
            gsl_permutation * p = gsl_permutation_alloc (eye->size1);
            gsl_linalg_LU_decomp (eye, p, &s);
            // Invert the matrix m
            gsl_matrix *work2 = gsl_matrix_calloc(Pmat->size1, Pmat->size2);
            gsl_linalg_LU_invert (eye, p, work2);
            gsl_matrix *work3 = gsl_matrix_alloc(Phi->size2, Phi->size1);
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, work, work2, 0, work3);
            gsl_matrix_scale(Pmat, 1.0/lambda);
            gsl_matrix_set_identity(eye);
            gsl_matrix_sub(eye, Pmat);
            //now eye is I - gamma P
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, work3, eye, 0, work);
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, work, Phi, 0, Amat);

            gsl_matrix_free(work2);
            gsl_matrix_free(work3);
            gsl_matrix_free(eye);
      }
      gsl_matrix_free(work);
}

void compute_bvec(double lambda, gsl_vector * b, const struct model_mdp_t * model_mdp){

      gsl_matrix *Phi = model_mdp->true_observations;
      gsl_vector *dmu = model_mdp->dmu;

      gsl_matrix *Pmat = gsl_matrix_calloc(model_mdp->numstates, model_mdp->numstates);
      compute_ppi_gamma(Pmat, model_mdp);
      // Pmat is actually gamma * P

      gsl_matrix *Dmat = gsl_matrix_calloc(dmu->size, dmu->size);
      gsl_matrix *work = gsl_matrix_calloc(Phi->size2, dmu->size);
      gsl_vector_view D_diag = gsl_matrix_diagonal (Dmat);
      gsl_vector_memcpy(&D_diag.vector, dmu);
      gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, Phi, Dmat, 0, work);

      gsl_vector *rewardpi = gsl_vector_calloc(model_mdp->numstates);
      compute_expected_reward(rewardpi, model_mdp);

      if (lambda == 0){
            gsl_blas_dgemv (CblasNoTrans, 1.0, work, rewardpi, 0, b);
      }
      else{
            gsl_matrix_scale(Pmat, lambda);
            gsl_matrix * eye = gsl_matrix_alloc(model_mdp->numstates, model_mdp->numstates);
            gsl_matrix_set_identity(eye);
            gsl_matrix_sub(eye, Pmat);
            //now eye is I - lambda gamma P
            // Make LU decomposition of matrix m
            int s = 0;
            gsl_permutation * p = gsl_permutation_alloc (eye->size1);
            gsl_linalg_LU_decomp (eye, p, &s);
            // Invert the matrix m
            gsl_matrix *work2 = gsl_matrix_calloc(Pmat->size1, Pmat->size2);
            gsl_linalg_LU_invert (eye, p, work2);
            gsl_matrix *work3 = gsl_matrix_alloc(Phi->size2, Phi->size1);
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, work, work2, 0, work3);

            gsl_blas_dgemv (CblasNoTrans, 1.0, work3, rewardpi, 0, b);
            //printf("reach comput b ------------\n");
            gsl_matrix_free(work2);
            gsl_matrix_free(work3);
            gsl_matrix_free(eye);
            gsl_permutation_free(p);
      }

      gsl_matrix_free(work);
      gsl_vector_free(rewardpi);
}

void compute_Cmat(gsl_matrix * matC, const struct model_mdp_t * model_mdp){

      gsl_matrix *Phi = model_mdp->true_observations;
      gsl_vector *dmu = model_mdp->dmu;

      gsl_matrix *work = gsl_matrix_calloc(Phi->size1, Phi->size1);
      gsl_vector_view diagw = gsl_matrix_diagonal(work);
      gsl_vector_memcpy(&diagw.vector, dmu);
      gsl_matrix *work1 = gsl_matrix_calloc(Phi->size2, Phi->size1);
      gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, Phi, work, 0, work1);
      gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, work1, Phi, 0, matC);

      int s = 0;
      gsl_permutation * p = gsl_permutation_alloc (matC->size1);
      gsl_linalg_LU_decomp (matC, p, &s);
      // Invert the matrix m
      gsl_matrix *work2 = gsl_matrix_calloc(matC->size1, matC->size2);
      gsl_linalg_LU_invert (matC, p, work2);
      gsl_matrix_memcpy(matC, work2);

      gsl_permutation_free(p);
      gsl_matrix_free(work);
      gsl_matrix_free(work1);
      gsl_matrix_free(work2);
}



/********************************************************/
