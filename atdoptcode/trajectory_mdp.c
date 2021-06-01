#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#include "trajectory_mdp.h"
#include "utils.h"

/*
 * Default trajectory_mdp
 */
// TODO: put in proper default file names here
// NOTE: numobservations is actually number of features
const struct trajectory_mdp_opts_t default_trajectory_mdp_opts =
{.numobservations = 2000, .trajectory_length = 2000, .num_true_states = 2000,
            .true_observations_filename = "mcarphi1k/evlvalues2000"};

/***** Private functions used below, in alphabetical order *****/

gsl_matrix * load_truevalues(struct trajectory_mdp_t * trajectory_mdp, const struct input_file_info * input_info);
int make_trajectory_mdp(struct trajectory_mdp_t * trajectory_mdp, const struct input_file_info * input_info);
void load_trajectory(struct trajectory_mdp_t * trajectory_mdp);

/***** End Private functions used below ************************/

void deallocate_trajectory_mdp_t(struct trajectory_mdp_t *trajectory_mdp){
      gsl_matrix_free(trajectory_mdp->Observations);
      gsl_matrix_free(trajectory_mdp->true_observations);
      gsl_vector_free(trajectory_mdp->true_values);
      gsl_vector_free(trajectory_mdp->rewards);
      gsl_vector_free(trajectory_mdp->gammas);
      gsl_vector_free(trajectory_mdp->rhos);
}


struct trajectory_mdp_t * get_trajectory_mdp(struct input_file_info *file_info) {
      //NOTE: filename is always NOT NULL when reading from files
      struct trajectory_mdp_t * trajectory_mdp = malloc(sizeof(struct trajectory_mdp_t));
      make_trajectory_mdp(trajectory_mdp, file_info);
      //printf("true value length is %d\n", (int)trajectory_mdp->true_values->size);
      return trajectory_mdp;
}

// Resets mdp back to the start of a run
void reset_trajectory_mdp(gsl_vector * x_tp1, struct trajectory_mdp_t *trajectory_mdp) {
      // Load the next trajectory and reset indexing to start
      trajectory_mdp->run++;
      trajectory_mdp->t = 0;

      int LEN = CHAR_BIT * sizeof(int)/3 + 2;
      char index[LEN];
      snprintf(index, LEN, "%d", trajectory_mdp->run);

      char *temp = trajectory_mdp->train_fileprefix;
      trajectory_mdp->train_filename = concat(temp, index);
      
      load_trajectory(trajectory_mdp);
      //there are totally numobservations + 2 columns
      gsl_vector_view rewardsvec = gsl_matrix_column(trajectory_mdp->Observations, trajectory_mdp->Observations->size2-2);
      gsl_vector_view gammasvec = gsl_matrix_column(trajectory_mdp->Observations, trajectory_mdp->Observations->size2-1);
      gsl_vector_memcpy(trajectory_mdp->gammas, &gammasvec.vector);
      gsl_vector_memcpy(trajectory_mdp->rewards, &rewardsvec.vector);
      // get first observation
      gsl_vector_view x = gsl_matrix_subrow(trajectory_mdp->Observations, trajectory_mdp->t, 0, trajectory_mdp->numobservations);
      //gsl_vector_print(&xrow.vector);
      if (trajectory_mdp->sparse == SPARSE) {
         gsl_vector_set_zero(x_tp1);
         for (int i = 0; i<trajectory_mdp->num_nonzeros; i++) {
             int changind = (int)gsl_vector_get(&x.vector, i);
            gsl_vector_set(x_tp1, changind, gsl_vector_get(x_tp1, changind)+1);
         }
      }
      else gsl_vector_memcpy(x_tp1, &x.vector);
}

void env_step_trajectory_mdp(gsl_vector * x_tp1, double * reward, double * gamma_tp1, double *rho_tm1, struct trajectory_mdp_t *trajectory_mdp) {
    
      trajectory_mdp->t++;
      // get the next state in the trajectory
      gsl_vector_view x = gsl_matrix_subrow(trajectory_mdp->Observations, trajectory_mdp->t, 0, trajectory_mdp->numobservations);
      //printf("----------------------------- %d\n", (int)(&x.vector)->size);
      if (trajectory_mdp->sparse == SPARSE) {
        gsl_vector_set_zero(x_tp1);
        for (int i = 0; i<trajectory_mdp->num_nonzeros; i++) {
            //printf(" %d ", (int)gsl_vector_get(&x.vector, i));
            int changind = (int)gsl_vector_get(&x.vector, i);
            gsl_vector_set(x_tp1, changind, gsl_vector_get(x_tp1, changind)+1);
        }
      }
      else gsl_vector_memcpy(x_tp1, &x.vector);
      //printf("the current reward is %f\n", (*gamma_tp1));
      *gamma_tp1 = gsl_vector_get(trajectory_mdp->gammas, trajectory_mdp->t);
      *rho_tm1 = gsl_vector_get(trajectory_mdp->rhos, trajectory_mdp->t);
      *reward = gsl_vector_get(trajectory_mdp->rewards, trajectory_mdp->t);
}

/************************ Private functions to create TRAJECTORY_MDP *********************/


int make_trajectory_mdp(struct trajectory_mdp_t * trajectory_mdp, const struct input_file_info * input_info) {
      if (input_info->sparse == SPARSE)
         trajectory_mdp->numobservations = input_info->num_nonzeros;
      else trajectory_mdp->numobservations = input_info->num_features;
    
      trajectory_mdp->train_fileprefix = input_info->trainfileprefix;
      trajectory_mdp->true_observations_filename = input_info->testfile;
    
      trajectory_mdp->sparse = input_info->sparse;
      trajectory_mdp->num_nonzeros = input_info->num_nonzeros;
      
      trajectory_mdp->true_values = gsl_vector_alloc(input_info->num_evl_states);;

      gsl_matrix *evlstatevalues = load_truevalues(trajectory_mdp, input_info);
      gsl_matrix_view temp = gsl_matrix_submatrix(evlstatevalues,0,0,input_info->num_evl_states, trajectory_mdp->numobservations); 
      //later this can be deleted after sparse computation developed, if sparse, recover to sparse representation
      trajectory_mdp->true_observations = gsl_matrix_alloc(input_info->num_evl_states, trajectory_mdp->numobservations);
      gsl_matrix_memcpy(trajectory_mdp->true_observations, &temp.matrix);
      
      gsl_vector_view truevaluescol = gsl_matrix_column(evlstatevalues, evlstatevalues->size2-1); 
      gsl_vector_memcpy(trajectory_mdp->true_values, &truevaluescol.vector);
      gsl_matrix_free(evlstatevalues);
      //gsl_vector_print(trajectory_mdp->true_values);
      // Newly created, so run = 0;
      trajectory_mdp->run = 0;
      trajectory_mdp->t = 0;
      int trajectory_len = input_info->train_length;
      // Initialize trajectory variables
      trajectory_mdp->rhos = gsl_vector_alloc(trajectory_len);
      trajectory_mdp->gammas = gsl_vector_alloc(trajectory_len);
      trajectory_mdp->rewards = gsl_vector_alloc(trajectory_len);
      trajectory_mdp->Observations = gsl_matrix_calloc(trajectory_len, trajectory_mdp->numobservations + 2);
      return 0;
}


// TODO: do not want to require that the save info be a gsl_matrix already
// want to generically take a csv; for now we can leave this
gsl_matrix * load_truevalues(struct trajectory_mdp_t * trajectory_mdp, const struct input_file_info * input_info) {
      
      FILE * true_observations_file = open_file(trajectory_mdp->true_observations_filename, "r");
      gsl_matrix * temp = gsl_matrix_alloc(input_info->num_evl_states, trajectory_mdp->numobservations+1);
      gsl_matrix_fread(true_observations_file, temp);

      fclose(true_observations_file);
      return temp;
}

// TODO: make this actually read in a file
// will have to add a string to trajectory_mdp_t definition to save
// file location information

// read in the trajectory data in this func, it will be called in the reset_trajectory_mdp function
void load_trajectory(struct trajectory_mdp_t * trajectory_mdp) {
      FILE * observations_file = fopen(trajectory_mdp->train_filename, "r");
      //printf("the traj file name is %s\n----------------------", trajectory_mdp->train_filename); 
      gsl_matrix_fread(observations_file, trajectory_mdp->Observations);
    
      gsl_vector_set_all(trajectory_mdp->rhos, 1.0);

      fclose(observations_file);
}
