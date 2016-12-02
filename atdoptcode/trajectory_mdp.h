#ifndef _TRAJECTORY_MDP_H
#define _TRAJECTORY_MDP_H

#define MAX_TRAJ_LEN 10000
#define SPARSE 0
#define NONSPARSE 1

#include "utils.h"

struct trajectory_mdp_t {
      int numobservations;

      // Trajectory of gammas, rewards, etc.
      gsl_vector * gammas;
      gsl_vector * rhos;
      gsl_vector * rewards;
      gsl_matrix * Observations;
      int t; // keep track of location in trajectory
      int run; // keep track of current run, and so which trajectory is currently in use

      // True values for error computation
      gsl_matrix * true_observations;
      gsl_vector * true_values;

      char * train_filename;
      char * train_fileprefix;
      char * true_observations_filename;
    
      int num_nonzeros;
    
      int sparse;
};

struct trajectory_mdp_opts_t {

      // this var is not neccessary if we don't use gsl_vector_fread
      int num_episodes;

      // Need dimension of observation and trajectory len, to read in saved trajectories
      int numobservations;
      int trajectory_length;    

      // Need number of true states from which rollouts were computed to read in saved values
      int num_true_states;  

      //char true_observations_filename[MAX_FILENAME_STRING_LENGTH];
      //char true_values_filename[MAX_FILENAME_STRING_LENGTH];

      char * true_observations_filename;
};

struct input_file_info {
    //number of features
    int num_features;
    //number of non zero entries in feature vector, if applicable
    int num_nonzeros;
    //number of samples in each trajectory
    int train_length;
    //number of states to do evaluation
    int num_evl_states;
    //specify sparse computation or not
    int sparse;
    
    char * trainfileprefix;
    char * testfile;
};


void deallocate_trajectory_mdp_t(struct trajectory_mdp_t *trajectory_mdp);

void env_step_trajectory_mdp(gsl_vector * x_tp1, double * reward, double * gamma_tp1, double *rho_tm1, struct trajectory_mdp_t *trajectory_mdp);

struct trajectory_mdp_t * get_trajectory_mdp(struct input_file_info *file_info);

void reset_trajectory_mdp(gsl_vector * x_tp1, struct trajectory_mdp_t *trajectory_mdp);

#endif
