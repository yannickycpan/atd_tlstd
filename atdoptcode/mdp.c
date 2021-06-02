#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "mdp.h"
#include "utils.h"

void create_mdp(struct mdp_t * mdp, int type, struct input_file_info *file_info) {
      /*
       * Check info in file to see if model_mdp; if so, load model_mdp info
       * Check info in file to see if trajectory_mdp; if so, load trajectory_mdp info
       * Can only be one or the other; the below functions return NULL if of incorrect type
       */
      mdp->sparse = NONSPARSE;
      // Define the functions required for each subclass
      if (type == 1) {
          //always pass in NULL, o.w. it does not work
            mdp->model_mdp = get_model_mdp(NULL);
            mdp->trajectory_mdp = NULL;
      } else {
            mdp->model_mdp = NULL;
            mdp->trajectory_mdp = get_trajectory_mdp(file_info);
      }
      if (type == 1 && mdp->model_mdp != NULL) {
            mdp->numobservations = mdp->model_mdp->numobservations;
            // Simply points to the same place; does not copy unnecessarily
            mdp->true_observations = mdp->model_mdp->true_observations;
            mdp->true_values = mdp->model_mdp->vstar;
            mdp->error_weighting = mdp->model_mdp->dmu;
            printf("--------------------make model mdp successfully\n");
      }
      else if (mdp->trajectory_mdp != NULL) {
            mdp->sparse = file_info->sparse;
            mdp->numobservations = file_info->num_features;
            mdp->true_observations = mdp->trajectory_mdp->true_observations;
            mdp->true_values = mdp->trajectory_mdp->true_values;
            mdp->error_weighting = gsl_vector_alloc(mdp->true_values->size);
            gsl_vector_set_all(mdp->error_weighting, 1.0);
            printf("--------------------make trajectory mdp successfully\n");
     }
}

void deallocate_mdp_t(struct mdp_t *mdp){
      if (mdp->model_mdp != NULL) {
            deallocate_model_mdp_t(mdp->model_mdp);
      }
      if (mdp->trajectory_mdp != NULL) {
            deallocate_trajectory_mdp_t(mdp->trajectory_mdp);
            gsl_vector_free(mdp->error_weighting);
      }
}

void init_transition_info(struct transition_info_t * tinfo, const struct mdp_t * mdp) {
      tinfo->x_t = gsl_vector_calloc(mdp->numobservations);
      tinfo->x_tp1 = gsl_vector_calloc(mdp->numobservations);
}

void deallocate_transition_info(struct transition_info_t * tinfo) {
      gsl_vector_free(tinfo->x_t);
      gsl_vector_free(tinfo->x_tp1);
}

//NOTE: currently on-policy is assumed, so the rho is always 1.0
void reset_mdp_and_transition_info(struct transition_info_t * tinfo, struct mdp_t * mdp) {
      // Start MDP at new run
      // Get x0, gamma_1, r_1, x_1
      // Here we should pass in x_tp1, since before learning, env_step will be called
      // otherwise x_t will be covered by zero
      if (mdp->model_mdp != NULL) {
            reset_model_mdp(tinfo->x_tp1, mdp->model_mdp);
      }
      else {
            reset_trajectory_mdp(tinfo->x_tp1, mdp->trajectory_mdp);
      }

      // gamma_t not used on the first step, so can set it however
      tinfo->gamma_tp1 = 0.0;
      tinfo->rho_tm1 = 1.0;
      // r_{t+1} and s_{t+1} will be set when env_step called for the first time
}


void env_step(struct transition_info_t * tinfo, const struct mdp_t * mdp) {

      // Previous x_tp1 is now x_t; as with gamma
      gsl_vector_memcpy(tinfo->x_t, tinfo->x_tp1);
      tinfo->gamma_t = tinfo->gamma_tp1;
      tinfo->rho_t = tinfo->rho_tm1;

      /* get the action */
      if (mdp->model_mdp != NULL) {
            env_step_model_mdp(tinfo->x_tp1, &(tinfo->reward), &(tinfo->gamma_tp1), mdp->model_mdp);
      } else {
            env_step_trajectory_mdp(tinfo->x_tp1, &(tinfo->reward), &(tinfo->gamma_tp1), &(tinfo->rho_tm1), mdp->trajectory_mdp);
      }
}

void get_mdp_Amat(const struct mdp_t * mdp, double lambda, gsl_matrix *Amat){
      if (mdp->model_mdp != NULL) {
            get_model_Amat(lambda, Amat, mdp->model_mdp);
      }
      else
            printf("A matrix unavailable!!!!!!!!! \n");
}

void get_mdp_Cmat(const struct mdp_t * mdp, gsl_matrix *Cmat){
      if (mdp->model_mdp != NULL) {
            get_model_Cmat(Cmat, mdp->model_mdp);
      }
      else
            printf("A matrix unavailable!!!!!!!!! \n");
}


