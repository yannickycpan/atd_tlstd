#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

#include "experiment_utils.h"


void usage(char * program_name) {
      printf("usage: ./%s num_runs num_steps mdp_type [optional] mdp_filename \n", program_name);
      printf("example: ./%s 100 2000 1 \n", program_name);
      printf("runs a random mdp (type 1) for 100 runs and 2000 steps. \n");
      printf("Type 2 would give a trajectory mdp. \n");
      exit(-1);
}


int main(int argc, char * argv[]) {

      if (argc < 3) {
            usage(argv[0]);
      }

      const int num_runs = atoi(argv[1]); /* number of repeats */
      const int num_steps = atoi(argv[2]);
      const int steps_per_err = 50;
      const int num_nz = 10;
      // Iterators used throughout
      int i, j; // generic iterators

      /* Algorithms parameter sets */
      const int numalpha = 13;
      double alphas_range[13];
      for(i = 0; i < numalpha; i++)
          alphas_range[i] = 0.1*pow(2,i-7);
      
      int numxi = 13;
      double xis_range[13];
      for(int i = 0; i<numxi; i++){
          xis_range[i] = 0.1*pow(2, -9+i);
      }

      //printf("alpha 0 is %lf\n", alphas_range[0]);
      const int numeta = 13;
      double etas_range[13];
      double etastep = 0;
      for(i = 0; i < numeta; i++){
        etas_range[i] = pow(10, -5 + etastep);
        etastep+=0.75;
      }

      double lambdas_range[] =  {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95,0.97,0.99,1.0};
      const int numlambda = sizeof(lambdas_range)/sizeof(double);

      // Initialize params array
      int num_params = numalpha*numlambda;
      struct alg_params_t alg_params[num_params];
      int pp = 0, pick_ind = 0, count = 0;
      int cluster_indicator = 0;
      double factor = 0.00001;
     
      if(argc > 4){
        pick_ind = atoi(argv[4]);
        cluster_indicator = 1;
      }

      for (i = 0; i < numalpha; i++) {
            for (j = 0; j < numlambda; j++) {
                  if(count == pick_ind || cluster_indicator == 0){
                  alg_params[pp].alpha_t = alphas_range[i];
                  alg_params[pp].lambda_t = lambdas_range[j];
                  alg_params[pp].beta_t = factor;
                  alg_params[pp].lambda_tp1 = lambdas_range[j];
                  alg_params[pp].eta_t = etas_range[i];
                  alg_params[pp].threshold = 0.01; 
                  pp++;
                  }
                  count++;
            }
      }
      num_params = pp;
      int paramstartone = 1;
      int paramstarttwo = num_params;

      // Initialize mdp struct, which will be reused across runs and mdps
      struct mdp_t mdp;
      struct input_file_info input_info;
      input_info.num_features = 1024;
      input_info.train_length = num_steps;
      input_info.num_evl_states = 2000;
      input_info.num_nonzeros = num_nz;
      //use SPARSE will use sparse computation, nonzero is 1 by default
      //NOTE: current sparse only means it allocate and read in nonzero indexes
      input_info.sparse = NONSPARSE;
      input_info.trainfileprefix = "mcarphi1k/mcartrainphi1k";
      input_info.testfile = "mcarphi1k/mcartestphi1k";
      create_mdp(&mdp, atoi(argv[3]), &input_info);


      /* Algorithms */
      const char *alg_names[] = {"TD","TO-TD","TO-ETD","ETD","ATD2nd","LSTD"};
      const int num_algs = sizeof(alg_names)/sizeof(char *);

      /* Run algorithms on given mdps */
      struct result_vars_t rvars;
      rvars.num_algs = num_algs;
      rvars.num_params = num_params;
      rvars.num_steps = num_steps;
      rvars.steps_per_err = steps_per_err;
      rvars.num_errsteps = num_steps/steps_per_err;
      rvars.num_runs = num_runs;
      allocate_result_vars(&rvars);
      printf("start running \n");
      run_exp(&rvars, &mdp, alg_names, num_algs, alg_params);


      /*
       * Print information to file
       */
      char full_prefix[MAX_FILENAME_STRING_LENGTH];
      char * directory_name = "RMDPresults";
      char * prefix = "OnPolicy";
      char * file_name = "boyanjob";
      memset(full_prefix, 0, sizeof(char)*MAX_FILENAME_STRING_LENGTH);
      if (argc > 4)
           sprintf(full_prefix,"%s/jobs/%s_%d", directory_name, file_name, atoi(argv[4]));
      else sprintf(full_prefix,"%s/%s_%d_to_%d",directory_name,prefix,paramstartone, paramstarttwo);

      // Ensure directory exists and then print results
      create_directory(directory_name);
      print_results_to_file(full_prefix, &rvars, alg_names, alg_params);

      deallocate_mdp_t(&mdp);
      deallocate_result_vars(&rvars);

      return 1;
}
