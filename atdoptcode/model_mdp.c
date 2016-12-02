#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "model_mdp.h"
#include "utils.h"

/*
 * Default model_mdp
 */
const struct model_mdp_opts_t default_model_mdp_opts = {.branch = 2, .gamma = 0.99, .numactions = 2, .numstates = 30,
            .pi_prob = 0.9, .mu_prob = 0.9, .offpolicy = 0,
            .observation_type = "tabular", .reward_type = "random_uniform", .seed = 0};


/***** Private functions used below, in alphabetical order *****/
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
      model_mdp->s_t = uniform_random_int_in_range(model_mdp->rgen, model_mdp->numstates);

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
      for (a = 0; a < opts->numactions; a++) 
            gsl_matrix_set_all(model_mdp->gammas[a],opts->gamma);

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
      get_dmu(model_mdp->dmu, model_mdp);

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
      for (a = 0; a < opts->numactions; a++)      
            gsl_matrix_set_all (model_mdp->rewards[a],0.0);

      a = 0;
      if (strcmp(opts->reward_type,"sparse")==0)
            gsl_matrix_set (model_mdp->rewards[a], model_mdp->numstates-2, model_mdp->numstates-1, 1.0);
      else if (strcmp(opts->reward_type,"const")==0)
            gsl_matrix_set_all (model_mdp->rewards[a], 1.0);
      else if (strcmp(opts->reward_type,"random_uniform")==0)
            generate_random_uniform_matrix(model_mdp->rewards[a], model_mdp->rgen,0);	

      for (a = 1; a < opts->numactions; a++)      
            gsl_matrix_memcpy (model_mdp->rewards[a],model_mdp->rewards[0]);

}

int generate_P_matrix(gsl_matrix * P [MAX_ACTIONS], const int na, const int branch, struct rgen_t *rgen){
      gsl_vector * next_st = gsl_vector_alloc(branch);
      int nextStates = branch;
      /* loop over actions */
      for (int i=0; i<na; i++) {
            gsl_matrix_set_all(P[i],0);

            for(int j=0;j<P[i]->size1;j++){
                  if(branch == P[i]->size2)
                  {
                        nextStates = uniform_random_int_in_range(rgen, P[i]->size2)+1;
                        next_st = gsl_vector_alloc(nextStates);

                  }
                  generate_random_indices(next_st, P[i]->size2, rgen);
                  for (int k = 0; k < next_st->size; k++) {
                        int index = gsl_vector_get(next_st,k);
                        double val = uniform_random(rgen);
                        if(val < MIN_VAL)
                              val = MIN_VAL;
                        gsl_matrix_set(P[i],j,index,val);
                  }
            }


            for (int j = 0; j < P[i]->size1; j++) {
                  double sum = 0.0;
                  for (int k = 0; k < P[i]->size2; k++) {
                        sum += gsl_matrix_get(P[i], j, k);
                  }
                  for (int k = 0; k < P[i]->size2; k++) {
                        gsl_matrix_set(P[i],j,k,gsl_matrix_get(P[i],j,k)/sum);
                  }
            }
      }
      gsl_vector_free(next_st);
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

      /* setup feature matrix */
      if (strcmp(opts->observation_type,"tabular")==0){
            model_mdp->numobservations = model_mdp->numstates;
            model_mdp->numobservations = model_mdp->numstates;
            model_mdp->true_observations = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numobservations);
            gsl_matrix_set_identity(model_mdp->true_observations);
      }
      else if (strcmp(opts->observation_type,"local")==0)
      {
            model_mdp->numobservations = model_mdp->numstates;
            model_mdp->true_observations = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numobservations);

            gsl_matrix_set_identity(model_mdp->true_observations);
            double v;
            int i, j, a;
            for (i = 0; i < model_mdp->numstates; i++)
            {
                  for (j = 0; j < model_mdp->numstates; j++){
                        v = 0.0;
                        // Check if there is any probability of transitioning
                        for(a = 0; a < model_mdp->numactions; a++)
                              v += gsl_matrix_get(model_mdp->P[a],i,j);

                        if(v > 0)
                              gsl_matrix_set(model_mdp->true_observations,i,j,1.0);
                  }
            }
            normalize_rows_matrix(model_mdp->true_observations);
      }
      else if (strcmp(opts->observation_type,"inverted")==0)
      {
            model_mdp->numobservations = model_mdp->numstates;
            model_mdp->true_observations = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numobservations);

            gsl_matrix_set_all(model_mdp->true_observations,1.0);
            for (int i=0;i<model_mdp->numstates;i++)
            {
                  gsl_matrix_set(model_mdp->true_observations,i,i,0.0);
            }
            normalize_rows_matrix(model_mdp->true_observations);
      }
      else if (strcmp(opts->observation_type,"binary")==0)
      {
            int nbits = floor(log2(model_mdp->numstates+1))+1;
            model_mdp->numobservations = nbits;
            model_mdp->true_observations = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numobservations);
            gsl_vector * vec = gsl_vector_alloc(nbits);
            for(int i=1;i<=model_mdp->numstates;i++){
                  int2Binary(vec, i,model_mdp->numstates+1);
                  gsl_matrix_set_row (model_mdp->true_observations, i-1, vec);
            }
            // Additionally, if binary_norm, then normalize these observations
            if (strcmp(opts->observation_type,"binary_norm")==0) {
                  normalize_rows_matrix(model_mdp->true_observations);
            }
      }
      else if  (strcmp(opts->observation_type,"random_normal")==0)
      {
            model_mdp->numobservations = min(model_mdp->numstates - 2, 1);
            model_mdp->true_observations = gsl_matrix_alloc(model_mdp->numstates,model_mdp->numobservations);

            gsl_matrix_set_zero (model_mdp->true_observations);
            for (int i=0; i<model_mdp->true_observations->size1; i++) {
                  for (int j=0; j<model_mdp->true_observations->size2-1; j++){
                        gsl_matrix_set (model_mdp->true_observations, i, j, uniform_random(model_mdp->rgen));
                  }
                  gsl_matrix_set (model_mdp->true_observations, i, model_mdp->true_observations->size2-1, 1.0);
            }
            normalize_rows_matrix(model_mdp->true_observations);
      }
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

      // Compute vstar as (I - Ppigammma)^{-1} Rbar
      int s;
      gsl_matrix * M = gsl_matrix_alloc (model_mdp->numstates, model_mdp->numstates);
      gsl_permutation * perm = gsl_permutation_alloc (model_mdp->numstates);
      /* Make LU decomposition of matrix eye */
      gsl_linalg_LU_decomp (eye, perm, &s);
      /* Invert the matrix eye put result in M */
      gsl_linalg_LU_invert (eye, perm, M);

      gsl_matrix_vector_product(vstar,M, expected_reward);

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


/********************************************************/
