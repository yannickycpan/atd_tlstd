
#ifndef _BOYAN_MDP_H
#define _BOYAN_MDP_H

#define MAX_STATES 100
#define MAX_FEATURES 100
#define MAX_ACTIONS 100
#define MIN_VAL 0.0002

#include "utils.h"

/*
 * tabular features, sigma = 0.1. gamma=0.99
 * if alpha is NOT decayed on each step a good alpha range seems to be -7:0
 * if alpha IS decayed on each step a good alpha range seems to be -4:3
 *
 * RANDOM features, sigma = 0.1. gamma=0.9
 * if alpha is NOT decayed on each step a good alpha range seems to be -9:-2
 * if alpha is IS decayed on each step a good alpha range seems to be -5:2
 *r
 * RANDOM features, sigma = .5 gamma=0.9
 * if alpha is NOT decayed on each step a good alpha range seems to be -9:-2
 * if alpha is IS decayed on each step a good alpha range seems to be -5:2
 *
 * TABULAR features, sigma = .1 gamma=0.7
 * if alpha is NOT decayed on each step a good alpha range seems to be -7:0
 * if alpha IS decayed on each step a good alpha range seems to be -4:3
 */

struct model_mdp_t {
      int s_t;
      int numobservations;

      // MDP definitions
      int numstates;
      int numactions;
      gsl_matrix * P [MAX_ACTIONS];
      gsl_matrix * gammas [MAX_ACTIONS];
      gsl_matrix * rewards [MAX_ACTIONS];
      gsl_matrix * true_observations;

      // Policies
      gsl_matrix * mu;
      gsl_matrix * pi;

      // Sampling probabilities; cumulative probabilities to make it faster to sample
      double A_cdf[MAX_STATES][MAX_ACTIONS];
      double SA_cdf[MAX_ACTIONS][MAX_STATES][MAX_STATES];

      struct rgen_t * rgen;

      // store true values to obtain errors
      gsl_vector * vstar;
      gsl_vector * dmu;

};

// Read mdp_opts from a file
struct model_mdp_opts_t {

      int branch; /* number of next possible states per state per action */
      double gamma; /* single constant gamma for now; generalize later to other types */

      int numactions; 
      int numstates;

      double pi_prob;
      double mu_prob;
      int offpolicy;  // If mu_prob != pi_prob, then this variable automatically set
      // Otherwise, even if mu_prob = pi_prob, could still be off-policy

      char observation_type[MAX_OPTION_STRING_LENGTH];
      char reward_type[MAX_OPTION_STRING_LENGTH];

      // Parameters for randomness
      unsigned long int seed;      
};


void deallocate_model_mdp_t(struct model_mdp_t *model_mdp);

void env_step_model_mdp(gsl_vector * x_tp1, double * reward, double * gamma_tp1, struct model_mdp_t *model_mdp);

struct model_mdp_t * get_model_mdp(const char * filename);

void reset_model_mdp(gsl_vector * x_0, struct model_mdp_t * model_mdp);

void get_model_Amat(double lambda, gsl_matrix * Amat, struct model_mdp_t *model_mdp);

void get_model_Cmat(gsl_matrix * matC, struct model_mdp_t *model_mdp);

#endif
