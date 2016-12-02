
#ifndef _MDP_H
#define _MDP_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_blas.h>
#include <time.h>

#include "rl_interface.h"
#include "boyan_mdp.h"
#include "trajectory_mdp.h"

/* 
 * MDP assumes that there are only two types of MDPs: those defined by a model,
 and those defined by a trajectory.
 */


// TODO: eventually consider including a few function pointers inside mdp_t or transition_info_t
// that generalically compute dot product on observation vectors
// to automatically enable sparse computation without knowing the type of x_t
struct mdp_t {
      // TODO: this currently refers to numfeatures, but should be changed to have features as part of agent
      // Called numobservations here as this will change soon
      int numobservations; 

      // Model info
      struct model_mdp_t * model_mdp;

      // Trajectory info
      struct trajectory_mdp_t * trajectory_mdp;

      // True values for error computation
      gsl_matrix * true_observations;
      gsl_vector * true_values;
      gsl_vector * error_weighting; // weights each error entry, e.g. dmu
      // Additional work values for error computation, of size numfeatures
      gsl_vector * work;
      
      int sparse;
      // Functions to get features for a state
};

void create_mdp(struct mdp_t * mdp, int type, struct input_file_info *file_info);

void deallocate_mdp_t(struct mdp_t *mdp);

void env_step(struct transition_info_t * tinfo, const struct mdp_t * mdp);

void init_mdp(struct mdp_t * mdp);

void init_transition_info(struct transition_info_t * tinfo, const struct mdp_t * mdp);

void deallocate_transition_info(struct transition_info_t * tinfo);

void reset_mdp_and_transition_info(struct transition_info_t * tinfo, struct mdp_t * mdp);

void get_mdp_Amat(const struct mdp_t * mdp, double lambda, gsl_matrix *Amat);

void get_mdp_Cmat(const struct mdp_t * mdp, gsl_matrix *Cmat);


#endif
