#define _XOPEN_SOURCE 500
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "experiment_utils.h"

/***** Private functions used below, in alphabetical order *****/

int load_algs(struct alg_t * algs[], const char *alg_names[], const int num_algs, const int numobservations);
void deallocate_algs(struct alg_t * algs[], const int num_algs);

/***** End Private functions used below ************************/


/***************************************************
 * Functions for running experiments
 ***************************************************/

void allocate_result_vars(struct result_vars_t * rvars) {

      const int total_size = rvars->num_algs*rvars->num_params*rvars->num_errsteps;
      rvars->mean_mse = calloc(total_size, sizeof(double *));
      rvars->var_mse = calloc(total_size, sizeof(double *));
      rvars->mean_time = calloc(total_size, sizeof(double *));
      rvars->var_time = calloc(total_size, sizeof(double *));

      rvars->array_sizes[ALG_IND] = rvars->num_algs;
      rvars->array_sizes[PARAM_IND] = rvars->num_params;
      rvars->array_sizes[STEP_IND] = rvars->num_errsteps;      
}

void deallocate_result_vars(struct result_vars_t * rvars) {
      free(rvars->mean_mse);
      free(rvars->var_mse);
      free(rvars->mean_time);
      free(rvars->var_time);
}


int run_exp(struct result_vars_t * rvars, struct mdp_t * mdp, const char *alg_names[], const int num_algs, const struct alg_params_t alg_params[]){

      // Iterators used throughout
      int aa; // iterator for actions
      int pp; // iterator for parameter setting
      int nn; // iterator for num_steps
      int index; // iterator for recording errors
      int indices[NUM_COMBOS_RESULTS]; // indices for getting the flattened index


      struct transition_info_t tinfo;
      init_transition_info(&tinfo, mdp);

      // Variables for computing errors and runtimes
      struct timeval tim;
      double t1;
      double t2;
      double err = 0.0;

      // Load algorithms
      struct alg_t * algs[num_algs];
      load_algs(algs, alg_names, num_algs, mdp->numobservations);
      // Work variable used in error computation repeatedly, so allocated here to save computation
      gsl_vector * work = gsl_vector_alloc(mdp->true_observations->size1);
      // Cycle through all parameters
      for (pp = 0; pp < rvars->num_params; pp++) {
            indices[PARAM_IND] = pp;
            // Set current parameters
            for (aa = 0; aa < rvars->num_algs; aa++) {
                  set_params(&(algs[aa]->params), &(alg_params[pp]));
            }
           if(mdp->trajectory_mdp != NULL)
               mdp->trajectory_mdp->run = 0;
           printf("current params are alpha, lambda: %f,%f\n", algs[0]->params.alpha_t, algs[0]->params.lambda_t);

            int z;
            for (z = 0; z < rvars->num_runs; z++) {
                  // Reset algorithms for new run
                  for (aa = 0; aa < rvars->num_algs; aa++) {
                        reset_alg(algs[aa]);
                  }
                  reset_mdp_and_transition_info(&tinfo, mdp);
                  
                  int err_count = 0;
                  indices[STEP_IND] = err_count;
                  for (aa = 0; aa < rvars->num_algs; aa++)  {
                      indices[ALG_IND] = aa;
                      err = get_rmse(algs[aa], mdp, work);
                      //printf("the current err is %f\n", err);
                      index = flatten_index(indices, rvars->array_sizes, NUM_COMBOS_RESULTS);
                      rvars->mean_mse[index] += err;
                      rvars->var_mse[index] += err*err;
                      rvars->mean_time[index] += 0;
                      rvars->var_time[index] += 0;
                  }
                  err_count++; 
                  for (nn = 1; nn < rvars->num_steps; nn++) {
                        indices[STEP_IND] = err_count;

                        env_step(&tinfo, mdp);
                        double time_temp[rvars->num_algs];
                        
                        // Update each algorithm
                        for (aa = 0; aa < rvars->num_algs; aa++)  {
                              //printf("the scalor is %f\n", scalor);
                              t1 = gettime(&tim);
                              algs[aa]->update_fcn(algs[aa]->alg_vars, &(algs[aa]->params), &tinfo);
                              t2 = gettime(&tim) - t1;
                              time_temp[aa] = t2;
                        }
                        /* Update errors for each alg */
                        if(nn % rvars->steps_per_err == 0){
                            for (aa = 0; aa < rvars->num_algs; aa++)  {
                                 indices[ALG_IND] = aa;
                                 err = get_rmse(algs[aa], mdp, work);
                                 if(nn % 100 == 0)printf("algo %s the current err is %f\n", algs[aa]->name, err);
                                 //if(nn % 100 == 0)printf("alg %s runtime is %f\n", algs[aa]->name, time_temp[aa]);
                                 index = flatten_index(indices, rvars->array_sizes, NUM_COMBOS_RESULTS);
                                 rvars->mean_mse[index] += err;
                                 rvars->var_mse[index] += err*err;
                                 rvars->mean_time[index] += time_temp[aa];
                                 rvars->var_time[index] += time_temp[aa]*time_temp[aa];
                            }
                            err_count++;
                        }

                  } /* loop over steps in trajectory */
            }/* loop over runs */

            /* after runs complete average data ------------- */
            for (aa = 0; aa < rvars->num_algs; aa++)  {
                  indices[ALG_IND] = aa;

                  for (nn = 0; nn < rvars->num_errsteps; nn++) {
                        indices[STEP_IND] = nn;

                        index = flatten_index(indices, rvars->array_sizes, NUM_COMBOS_RESULTS);

                        rvars->mean_mse[index] = rvars->mean_mse[index] / ((double) rvars->num_runs);
                        rvars->var_mse[index] = (rvars->var_mse[index]) / ((double) rvars->num_runs) - pow(rvars->mean_mse[index], 2);
                        rvars->mean_time[index] = rvars->mean_time[index] / ((double) rvars->num_runs);
                        rvars->var_time[index] = (rvars->var_time[index]) / ((double) rvars->num_runs) - pow(rvars->mean_time[index], 2);
                  }
            }
      }/* loop over parameters */

      gsl_vector_free(work);
      deallocate_transition_info(&tinfo);
      deallocate_algs(algs, num_algs);
      //printf("finish ---------------------  run experiments\n");
      return 1;
}


/***************************************************
 * Functions for printing results to file 
 ***************************************************/

// Output file suffixes
const char * RESULT_SUFFIXES[NUM_FILE_OUTPUTS] = {"PramNames", "LC", "Var"};

void remove_directory(char * directory_name){
      FILE *fp;
      int status;
      char * raw_command="rm -rf";
      char command[strlen(raw_command)+strlen(directory_name)+5];
      memset(command, 0, sizeof(command));
      sprintf(command,"%s %s",raw_command,directory_name);
      fp = popen(command, "r");
      status = pclose(fp);
      if (status == -1){
            //One could print a message here
      }
}

void create_directory(char * directory_name) {
      // Check first if the directory exists; if it does, do not create it
      struct stat st = {0};

      if (stat(directory_name, &st) == -1) {
            mkdir(directory_name, 0700);
      }
}


void print_results_to_file(const char * filename_prefix, const struct result_vars_t * rvars, const char * alg_names[], const struct alg_params_t alg_params[]) {

      char filenames[NUM_FILE_OUTPUTS][MAX_FILENAME_STRING_LENGTH];
      char param_string[MAX_PARAM_STRING_LENGTH];
      FILE * output_files[NUM_FILE_OUTPUTS];

      int indices[NUM_COMBOS_RESULTS];

      int i, aa, pp, nn, index;
      for(aa=0; aa < rvars->num_algs ; aa++){
            indices[ALG_IND] = aa;
            for (i = 0; i < NUM_FILE_OUTPUTS; i++) {
                  memset(filenames[i], 0, sizeof(char)*MAX_FILENAME_STRING_LENGTH);
                  sprintf(filenames[i],"%s_%s_%s.txt",filename_prefix,alg_names[aa],RESULT_SUFFIXES[i]);
                  // open_file throws error if files not opened
                  output_files[i] = open_file(filenames[i], "w+");
            }

            for (pp = 0; pp < rvars->num_params; pp++) {
                  indices[PARAM_IND] = pp;
                  fprintf(output_files[RESULT_SUFFIX_PRAM], "%s\n",params_to_string(param_string, &(alg_params[pp])));
                  for (nn = 0; nn < rvars->num_errsteps;nn++)
                  {
                        indices[STEP_IND] = nn;
                        index = flatten_index(indices, rvars->array_sizes, NUM_COMBOS_RESULTS);

                        if (nn == 0) {
                              fprintf(output_files[RESULT_SUFFIX_LC], "%f", rvars->mean_mse[index]);
                              fprintf(output_files[RESULT_SUFFIX_VAR], "%f", rvars->var_mse[index]);
                        } else {
                              fprintf(output_files[RESULT_SUFFIX_LC], ",%f", rvars->mean_mse[index]);
                              fprintf(output_files[RESULT_SUFFIX_VAR], ",%f", rvars->var_mse[index]);
                        }
                  }
                  fprintf(output_files[RESULT_SUFFIX_LC], "\n");
                  fprintf(output_files[RESULT_SUFFIX_VAR], "\n");
            }
      }
}



/***************************************************
 * PRIVATE Functions for loading algorithms
 ***************************************************/

int load_algs(struct alg_t * algs[], const char *alg_names[], const int num_algs, const int numobservations) {

      // Initialize algorithms
      int i, aa;
      int num_totalalgs = (sizeof(algorithm_map) / sizeof(algorithm_map[0]));
      int num_algsloaded = 0;
      for (aa = 0; aa < num_algs; aa++) {
            // Get the update function for this algorithm
            for (i = 0; i < num_totalalgs; i++) {
                  if (!strcmp(algorithm_map[i].name, alg_names[aa])) {
                        algs[aa] = calloc(1, sizeof(struct alg_t));
                        init_alg(algs[aa], algorithm_map[i].init_fcn, numobservations, alg_names[aa]);
                        break;
                  }
            }

            if (i == num_totalalgs)
                  fprintf(stderr, "experiment_utils-> algorithm %s not found!\n", alg_names[aa]);
            else
                  num_algsloaded++;
      }
      return num_algsloaded;
}

void deallocate_algs(struct alg_t * algs[], const int num_algs){
      int i, aa;
      int num_totalalgs = (sizeof(algorithm_map) / sizeof(algorithm_map[0]));
      for (aa = 0; aa < num_algs; aa++) {
            // Get the update function for this algorithm
            for (i = 0; i < num_totalalgs; i++) {
                  if (!strcmp(algorithm_map[i].name, algs[aa]->name)) {
                        deallocate_alg(algs[aa], algorithm_map[i].deallocate_fcn);
                        free(algs[aa]);
                        break;
                  }
            }
      }
}


double get_abse(const struct alg_t * alg, const struct mdp_t * mdp, gsl_vector * work) {

      double err = 0.0;

      alg->get_values(work, mdp->true_observations, alg->alg_vars);
      //gsl_matrix_vector_product(work, mdp->true_observations, w);

      gsl_vector_sub(work, mdp->true_values);
      gsl_vector_mul(work, mdp->error_weighting);

      // Absolute value error
      err = gsl_blas_dasum(work);

      return err;
}

// compute root mean squared error
// note if sparse, then the get true value function is not in use
double get_rmse(const struct alg_t * alg, const struct mdp_t * mdp, gsl_vector * work) {

      double err = 0.0;
      
      if(mdp->sparse == SPARSE) {
         struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg->alg_vars;
         gsl_blas_dgespmv(mdp->true_observations, vars->w, work); 
      }  
      else alg->get_values(work, mdp->true_observations, alg->alg_vars);

      gsl_vector_sub(work, mdp->true_values);
      //gsl_vector_print(mdp->true_values);
      for (int i = 0; i < work->size; i++) {
            gsl_vector_set(work, i, pow(gsl_vector_get(work, i), 2));
      }

      gsl_vector_mul(work, mdp->error_weighting);

      err = sqrt(gsl_blas_dasum (work)/(double)work->size);

      return err;
}

// compute percentage absolute mean error
// note should make sum of error weighting = 1
double get_pame(const struct alg_t * alg, const struct mdp_t * mdp, gsl_vector * work) {

      double err = 0.0;
    
      alg->get_values(work, mdp->true_observations, alg->alg_vars);

      gsl_vector_div(work, mdp->true_values);
      
      gsl_vector_add_constant(work, -1);

      err = gsl_blas_dasum (work);

      err = err/(double)work->size;

      return err;
}
