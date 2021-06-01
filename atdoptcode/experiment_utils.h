#ifndef _EXPERIMENT_UTILS_H
#define _EXPERIMENT_UTILS_H

// Define variables for storing results for num_algs x num_params x num_steps
// Note: this may seem to specific, but almost all RL experiments contain exactly this combination
#define NUM_COMBOS_RESULTS 3
#define ALG_IND 0
#define PARAM_IND 1
#define STEP_IND 2
// This is coincidentally also 3, corresponds to output files for params, mean and variance
#define NUM_FILE_OUTPUTS 3
#define RESULT_SUFFIX_PRAM 0
#define RESULT_SUFFIX_LC 1
#define RESULT_SUFFIX_VAR 2

#include "algorithms/all_algorithms.h"
#include "mdp.h"

struct result_vars_t {
      int num_runs;

      int num_algs;
      int num_params;
      int num_steps;
      int steps_per_err; 
      int num_errsteps;     
      int array_sizes[NUM_COMBOS_RESULTS];

      // Size of these variales is num_steps x num_params x num_algs
      double *mean_mse;
      double *var_mse;
      double *mean_time;
      double *var_time;
};

void allocate_result_vars(struct result_vars_t * rvars);

void deallocate_result_vars(struct result_vars_t * rvars);

int run_exp(struct result_vars_t * rvars, struct mdp_t * mdp, const char *alg_names[], const int num_algs, const struct alg_params_t alg_params[]);

void remove_directory(char * directory_name);

void create_directory(char * directory_name);

void print_results_to_file(const char * filename_prefix, const struct result_vars_t * rvars, const char * alg_names[], const struct alg_params_t alg_params[]);

double get_abse(const struct alg_t * alg, const struct mdp_t * mdp, gsl_vector * work);

double get_rmse(const struct alg_t * alg, const struct mdp_t * mdp, gsl_vector * work);

double get_pame(const struct alg_t * alg, const struct mdp_t * mdp, gsl_vector * work);
/********** below is the code for mapping algorithms **************/
// Mapping structure, map an algorithm name to its corresponding functions
const static struct{
      const char *name;
      AlgInitFcn init_fcn;
      AlgDeallocateFcn deallocate_fcn;
} algorithm_map [] = {
            { "TD", init_linear_alg, deallocate_linear_alg},
            { "TO-TD", init_linear_alg, deallocate_linear_alg},
            { "ETD", init_linear_alg, deallocate_linear_alg},
            { "TO-ETD", init_linear_alg, deallocate_linear_alg},
            { "TLSTD", init_matrix_alg, deallocate_matrix_alg},
            { "ATD2nd", init_matrix_alg, deallocate_matrix_alg},
            { "ATD2nd-FullA", init_matrix_alg, deallocate_matrix_alg},
            { "ATD2nd-TrueA", init_matrix_alg, deallocate_matrix_alg},
            { "LSTD", init_matrix_alg, deallocate_matrix_alg},
            { "TEST", init_sketch_alg, deallocate_sketch_alg},
            { "RPLSTD", init_sketch_alg, deallocate_sketch_alg},
};

#endif
