#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

#include "algorithms.h"
#include "domain_mdp.h"
#include "utils.h"

void usage()
{
      /* This function currently not used, as no parameters needed */
      printf("No parameters necessary, simpy run\n./compare_algorithms\n");
      exit(-1);
}

char* concat(char *s1, char *s2)
{
      char *result = malloc(strlen(s1)+strlen(s2)+1);//+1 for the zero-terminator
      //in real code you would check for errors in malloc here
      strcpy(result, s1);
      strcat(result, s2);
      return result;
}

int main(int argc, char * argv[])
{
      /* General experiment settings */
      const int numMDP = 1; /* number of random MDPs to run, length of krange */
      //const int N = 1; /* number of states in MDP */
      const int NS = 40; /* number of episodes */
      const int numRuns = 30; /* number of repeats */
      const int biasunit = 0;
      const int numallstates = 2000;
      char * trainfileprefix = "mcphidir1024fs20rand2k/";
      const int num_rand_fs = 0;
      const int num_feature = 1024 + num_rand_fs; 
      const int num_rand_nz = 0;
      const int num_nz = 10 + num_rand_nz;
      const int totalsteps = 5000;
      const int numerrorsample = 50;
      const int totalerrsteps = floor(totalsteps / ((double)numerrorsample));

      /* Algorithms options */
      double alphadiv = num_nz;
     
      int numalpha = 13;
      //NOTE: CANNOT use numalpha to INIT alphas_range
      double alphas_range[13];
      for(int i = 0; i<numalpha; i++){
          alphas_range[i] = 0.1*pow(2,i-7)/alphadiv;
      } 

      int numeta = 13;
      double etas_range[13];
      double etastep = 0;
      for(int i = 0; i<numeta; i++){
          etas_range[i] = pow(10, -4 + etastep);
          etastep += 0.75;
      }

      int numxi = 15;
      double xis_range[15];
      for(int i = 0; i<numxi; i++){
          xis_range[i] = 0.1*pow(2, -12+i);
      }

      double lambdas_range[15] =  {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93,0.95,0.97,0.99,1.0};
      int numlambda = sizeof(lambdas_range)/sizeof(double);

      int inputranks_range[1]= {50};
      const int numinputranks = sizeof(inputranks_range)/sizeof(int);

      printf("argc: %d\n", argc);
      // If argument passed, then only selecting one alpha
      if (argc > 1) {
         int i, j;
         int current_ind = 0;
         int pick_ind = atoi(argv[1]);
         for (i = 0; i < numalpha; i++) {
            for (j = 0; j < numlambda; j++) {
               if (current_ind == pick_ind) {
                  alphas_range[0] = alphas_range[i];
                  etas_range[0] = etas_range[i];
                  lambdas_range[0] = lambdas_range[j];
                  xis_range[0] = xis_range[i];
               }
               current_ind++;
            }
         }
         if (current_ind > numalpha*numlambda) {
            fprintf(stderr, "Index %d to parameters outside of range %d\n", pick_ind, numalpha*numlambda);
            exit (EXIT_FAILURE);
         }
         numalpha = 1;
         numlambda = 1;
         printf("alpha is %lf\n", alphas_range[0]);
         printf("lambda is %lf\n",lambdas_range[0]);
      } else {
         printf("numalpha is %d\n", numalpha);
         printf("numlambda is %d\n",numlambda);
      }
    
      double betas_range[1] = {0.5/8.0};
      const int numbeta = sizeof(betas_range)/sizeof(double); // since on-policy, ignore beta parameter
      //const int decay_alpha = 0;
      printf("numinputrank is %d\n",numinputranks);
      printf("numruns is %d\n",numRuns);

      const int nalgs = 9;
      const char *names[nalgs];
      const int TD_IND = 0;
      const int TO_TD_IND = 1;
      const int TO_ETD_IND = 2;
      const int ATD2nd_IND = 3;
      const int TLSTD_IND = 4;
      const int FLSTD_SA_IND = 5;
      const int LSTD_IND = 6;
      const int ILSTD_IND = 7;
      const int PLSTD_IND = 8;

      names[TD_IND] = "TD";
      names[TO_TD_IND] = "TO-TD";
      names[TO_ETD_IND] = "TO-ETD";
      names[FLSTD_SA_IND] = "FLSTD-SA";
      names[ATD2nd_IND] = "ATD2nd";
      names[TLSTD_IND] = "TLSTD";
      names[LSTD_IND] = "LSTD";
      names[ILSTD_IND] = "ILSTD";
      names[PLSTD_IND] = "PLSTD";
    
      int numofskt = 5;
   
      /* Define array to record running time for each algorithm, then define time struct*/
      struct timeval tim;
      double t1;
      double t2;
      double runningtime[nalgs];
      
      for(int xx=0;xx<nalgs;xx++){
         runningtime[xx] = 0;
      }
	  
      /* Used to compute statistics for average error and standard deviation */
      struct ArraySize sizes;
      sizes.n1 = nalgs;
      sizes.n2 = numalpha;
      sizes.n3 = numbeta;
      sizes.n4 = numlambda;
      sizes.n5 = totalerrsteps;
      sizes.n6 = numinputranks;
	
      //--------static allocations------
      const int totalSize = nalgs*numalpha*numbeta*numlambda*totalerrsteps*numinputranks;

      double *mse_gtd = malloc(totalSize * sizeof *mse_gtd);
      double *var_mse_gtd = malloc(totalSize * sizeof *var_mse_gtd);
      double *mean_mse_gtd = malloc(totalSize * sizeof *mean_mse_gtd);
      double *M2_mse_gtd = malloc(totalSize * sizeof *M2_mse_gtd);

      double *mse_gtd_overall = malloc(totalSize * sizeof *mse_gtd_overall);
      double *var_mse_gtd_overall = malloc(totalSize * sizeof *var_mse_gtd_overall);
      memset (mse_gtd_overall, 0, totalSize*sizeof(double));
      memset (var_mse_gtd_overall, 0, totalSize*sizeof(double));

      struct domain_mdp_t MDP;
      //these are used to generate random fs
      gsl_rng * rdfs = gsl_rng_alloc (gsl_rng_taus);
      int shuffle_fs[num_rand_fs];
      for(int i = 0; i<num_rand_fs; i++)shuffle_fs[i] = i + num_feature - num_rand_fs;

      //read in states and values used as test data
      char * allstatesfile = concat(trainfileprefix, "evlstates");
      char * truevaluefile = concat(trainfileprefix, "evlvalues2000");
      char * allstatesfile1indexes = concat(trainfileprefix, "evlstates1inds");
      FILE *allstates1indexes = fopen(allstatesfile1indexes, "r");
      gsl_matrix * allstates1indmat = gsl_matrix_alloc(numallstates, num_nz + biasunit);
      if (biasunit == 1) {
         gsl_matrix_view suball1indmat = gsl_matrix_submatrix(allstates1indmat, 0, 0, allstates1indmat->size1, allstates1indmat->size2 - 1);
         gsl_matrix_fread(allstates1indexes, &suball1indmat.matrix);
         gsl_vector_view unitcolall = gsl_matrix_column(allstates1indmat, allstates1indmat->size2 - 1);
         gsl_vector_set_all(&unitcolall.vector, num_feature);
      }
      //decide whether attach random features
      else if(num_rand_nz == 0)
         gsl_matrix_fread(allstates1indexes, allstates1indmat);
      else {
         gsl_matrix_view suball1indmat = gsl_matrix_submatrix(allstates1indmat, 0, 0, allstates1indmat->size1, allstates1indmat->size2 - num_rand_nz);
         gsl_matrix_fread(allstates1indexes, &suball1indmat.matrix);
         gsl_ran_shuffle(rdfs, shuffle_fs, num_rand_fs, sizeof(int));
         gsl_matrix_view subrand1indmat = gsl_matrix_submatrix(allstates1indmat, 0, num_nz - num_rand_nz, allstates1indmat->size1, num_rand_nz);
         for(int i = 0; i< allstates1indmat->size1; i++){
             gsl_vector_view oneindsrow = gsl_matrix_row(&subrand1indmat.matrix, i);
             for (int k = 0; k<num_rand_nz; k++) {
              gsl_vector_set(&oneindsrow.vector, k, shuffle_fs[k]);
             }      
         }
      }
      fclose(allstates1indexes);

      /*
       * Vectors that can be allocated now and re-used
       * Must be de-allocated later; if add one here, go de-allocate below
       */
    
      int k;
      for (k = 0; k < numMDP; k++) {

         load_truevalues(&MDP, num_feature, biasunit, numallstates, allstatesfile, truevaluefile);

         /*--------------------- store best performance for each MDP */
         memset (mse_gtd, 0, totalSize*sizeof(double));
         memset (var_mse_gtd, 0, totalSize*sizeof(double));
         memset (mean_mse_gtd, 0, totalSize*sizeof(double));
         memset (M2_mse_gtd, 0, totalSize*sizeof(double));
         /*--------------------- store best performance for each MDP */

         //mult will store estimated values for all states
         gsl_vector * mult = gsl_vector_alloc(numallstates);
         gsl_vector * x_t = gsl_vector_alloc(num_feature + biasunit);
         gsl_vector * x_t_1s = gsl_vector_alloc(num_nz + biasunit);
            
         gsl_vector * w_gtd[nalgs];
         alloc_vectors(w_gtd, nalgs, num_feature + biasunit);
         gsl_vector * e_gtd[nalgs];
         alloc_vectors(e_gtd, nalgs, num_feature + biasunit);
            
                
         int as;
         int bs = 0; // no beta values to iterate over
         int rs;
         for (bs = 0; bs < numbeta; bs++) {
            // Ignore beta for now, as on-policy
            for (rs = 0; rs < numinputranks; rs++) {
               int inputrank = inputranks_range[rs];
            
               for (as =0; as < numalpha; as++) {
                  double alpha = alphas_range[as];
                  double eta = etas_range[as];
                  double xi = xis_range[as];
                  int la;
                  for (la = 0; la < numlambda; la++)
                  {
                     double lambda = lambdas_range[la];
                     printf("Running (alpha, lambda) = (%f, %f)\n", alpha, lambda);
                     int z; //num of runs
                     int episodes = 0;
                     for (z = 0; z < numRuns; z++, episodes += NS)
                     {
                        int xx;
                        for(xx = 0;xx < nalgs; xx++){
                           gsl_vector_set_zero(w_gtd[xx]);
                           gsl_vector_set_zero(e_gtd[xx]);
                        }

                        /* For True-online base line algorithm */
                        double TOTD_VofS=0;
                        
                        // For ETD category algorithms
                        double toetdF = 0, toetdD = 0;
                        gsl_vector *toetd_oldw = gsl_vector_calloc(1);
                        
                        // init rank and step counter 
                        int r = inputrank;
                        int change_r = 0;
                        
                        gsl_matrix * matu = gsl_matrix_calloc(x_t->size,1);
                        gsl_vector * vecs = gsl_vector_calloc(1);
                        gsl_matrix * matv = gsl_matrix_calloc(x_t->size,1);
                        gsl_matrix * matl = gsl_matrix_alloc(1,1);
                        gsl_matrix * matr = gsl_matrix_alloc(1,1);
                        gsl_matrix_set_identity(matl);
                        gsl_matrix_set_identity(matr);
                        gsl_vector * b = gsl_vector_calloc(x_t->size);
                        
                        // For ilstd algorithm
                        gsl_matrix *ilstd_a = gsl_matrix_calloc(x_t->size, x_t->size);
                        gsl_vector *mu = gsl_vector_calloc(x_t->size);
                        
                        int shuffle[50-1];
                        for(int i = 0; i<inputrank-1; i++)shuffle[i] = i;
                        gsl_rng * rd = gsl_rng_alloc (gsl_rng_taus);                       
                        
                        // for sherman LSTD 
                        gsl_matrix *lstda = gsl_matrix_calloc(x_t->size, x_t->size);
                        gsl_vector *lstdb = gsl_vector_calloc(x_t->size);
                        gsl_vector_view diaglstda = gsl_matrix_diagonal(lstda);
                        gsl_vector_set_all(&diaglstda.vector, eta);
                        
                        // init for random projection
                        int plstdr = inputrank;
                        gsl_matrix *skt[numofskt];
                        struct rgen_t rt;
                        const gsl_rng_type * T;
                        T = gsl_rng_default;
                        rt.r = gsl_rng_alloc (T);
                        gsl_matrix * pa[numofskt];
                        gsl_vector * plstdb[numofskt];
                        gsl_vector * pz[numofskt];
                        gsl_vector * pw[numofskt];
                        gsl_matrix * palls[numofskt];
                        for(int i = 0; i<numofskt; i++){
                          palls[i] = gsl_matrix_calloc(numallstates, plstdr); 
                          pa[i] = gsl_matrix_calloc(plstdr, plstdr);
                          gsl_matrix_set_identity(pa[i]);
                          gsl_matrix_scale(pa[i], eta);
                          pz[i] = gsl_vector_calloc(plstdr);
                          pw[i] = gsl_vector_alloc(plstdr);
                          plstdb[i] = gsl_vector_calloc(plstdr);
                          skt[i] = gsl_matrix_calloc(x_t->size, plstdr);
                          generate_random_matrix(skt[i], 1.0/sqrt((double)plstdr), 0.0, &rt);
                          gsl_blas_dgespmm(MDP.Allstates, allstates1indmat, skt[i], palls[i]);
                        }
                        
                        int i;
                        //this episode index is for ilstd stepsize
                        int episodeind = 1;
                        int steps = 0, errsteps = 0;
                        /* compute error before learning */
                        for(xx=0; xx<nalgs;xx++){
                            gsl_blas_dgespmv(MDP.Allstates, allstates1indmat, w_gtd[xx], mult);
                            double err = compute_rmse(mult, MDP.Alltruevalues);
                            mse_gtd[SixDto1DIndex(xx,as,bs,la,errsteps,rs,&sizes)] += err;
                            var_mse_gtd[SixDto1DIndex(xx,as,bs,la,errsteps,rs,&sizes)] += err*err;
                        }

                        errsteps++;
                         
                        for (i = 0 + episodes; i < episodes + NS; i++) {
 
                           if(steps >= totalsteps)continue;
          
                           int LEN = CHAR_BIT * sizeof(int)/3 + 2;
                           char index[LEN];
                           snprintf(index, LEN, "%d", i);
                           char * phifilename = concat(trainfileprefix, concat("mcphi",index));
                           char * rewardfilename = concat("reward",index);
                           //no reward file here
                            
                           char * stepsfile = concat(trainfileprefix, "steps_episode");
                           gsl_vector * stepsperepi = gsl_vector_alloc(10000);
                           FILE * steps_episode = fopen(stepsfile,"r");
                           gsl_vector_fread(steps_episode, stepsperepi);
                           fclose(steps_episode);
                           int epi_len = (int)gsl_vector_get(stepsperepi, i);
                           gsl_vector_free(stepsperepi);
                           getDomainMDP(&MDP, num_feature, biasunit, epi_len, phifilename, rewardfilename);
                            
                           //read sparse matrices indexes
                           char * onesindexesfilename = concat(trainfileprefix, concat("mcphiones",index));
                           FILE * onesindexesfile = fopen(onesindexesfilename,"r");
                           gsl_matrix * onesind = gsl_matrix_alloc(epi_len, num_nz + biasunit);
                           if (biasunit == 1) {
                              gsl_matrix_view sub1indmat = gsl_matrix_submatrix(onesind, 0, 0, onesind->size1, onesind->size2 - 1);
                              gsl_matrix_fread(onesindexesfile, &sub1indmat.matrix);
                              gsl_vector_view unitcol = gsl_matrix_column(onesind, onesind->size2 - 1);
                              gsl_vector_set_all(&unitcol.vector, num_feature);
                           }
                           else if(num_rand_fs == 0)
                               gsl_matrix_fread(onesindexesfile, onesind);
                           else {
                              gsl_matrix_view sub1indmat = gsl_matrix_submatrix(onesind, 0, 0, epi_len, num_nz - num_rand_nz);
                              gsl_matrix_fread(onesindexesfile, &sub1indmat.matrix);
                              gsl_ran_shuffle(rdfs, shuffle_fs, num_rand_fs, sizeof(int));
                              gsl_matrix_view subrand1indmat = gsl_matrix_submatrix(onesind,0,num_nz-num_rand_nz,epi_len,num_rand_nz);
                              for(int i = 0; i< onesind->size1; i++){
                                  gsl_vector_view oneindsrow = gsl_matrix_row(&subrand1indmat.matrix, i);
                                  for (int k = 0; k<num_rand_nz; k++)
                                      gsl_vector_set(&oneindsrow.vector, k, shuffle_fs[k]);
                              }
                           }                    
                           fclose(onesindexesfile);
                            
                           InitPHI(MDP.PHI, onesind);
                            
                           // Initialize state, gamma, etc.
                           gsl_vector_view x = gsl_matrix_row(MDP.PHI, 0);
                           gsl_vector_memcpy(x_t, &x.vector);
                           gsl_vector_view x_t_1s_view = gsl_matrix_row(onesind, 0);
                           gsl_vector_memcpy(x_t_1s, &x_t_1s_view.vector);
                           
                           /******* init vector variables for each algorithm ************/
                           gsl_blas_spddot(w_gtd[TO_TD_IND], x_t, x_t_1s, &TOTD_VofS); 
                            
                           for(xx = 0;xx<nalgs;xx++){
                              gsl_vector_set_zero(e_gtd[xx]);
                           }
                            
                           for(int ss = 0;ss<numofskt;ss++){
                              gsl_vector_set_zero(pz[ss]);
                           }

                           gsl_vector_set_zero(mu);
                           double gamma_t = 0, gamma_tp1 = 1;
                           double reward_val = -1;
                           //one episode start
                           //printf("The current episode length is: %d\n", epi_len);
                           int updated_svd = 0, j = 1;
                           for (j=1; j<epi_len && steps<totalsteps; j++) {

                              double alpha_t = alpha;
                              gsl_vector_view x_tp1 = gsl_matrix_row(MDP.PHI, j);
                              gsl_vector_view x_tp1_1s = gsl_matrix_row(onesind, j);
 
                              // IMPORTNAT: updated_svd set to zero for each new sample
                              updated_svd = 0;
                              if (j == epi_len - 1) {
                                 gamma_tp1 = 0;
                              }
                           gettimeofday(&tim, NULL);
                           t1 = tim.tv_sec+(tim.tv_usec/1000000.0); 
                           if((change_r+1)%r == 0 && j >= r){
                              gsl_ran_shuffle(rd, shuffle, plstdr, sizeof(int));
                              for(int ii = 0; ii < r-1; ii++){
                              int tupleindex = (j-r) + shuffle[ii];  
                              double gamma_tp1_td = gamma_tp1;
                              if(tupleindex == epi_len - 1)gamma_tp1_td = 0;
                              gsl_vector_view x_td = gsl_matrix_row(MDP.PHI, tupleindex);
                              gsl_vector_view x_td_1s = gsl_matrix_row(onesind, tupleindex);
                              gsl_vector_view x_tp_td = gsl_matrix_row(MDP.PHI, tupleindex+1);
                              gsl_vector_view x_tp_td_1s = gsl_matrix_row(onesind, tupleindex+1);
                              TD_lambda_sp(w_gtd[FLSTD_SA_IND], e_gtd[FLSTD_SA_IND], 0, alpha_t, reward_val, gamma_t, gamma_tp1_td, &x_td.vector, &x_tp_td.vector, &x_td_1s.vector, &x_tp_td_1s.vector);
                              }
                            }
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[FLSTD_SA_IND] += t2-t1;

                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            TO_TD_lambda_sp(w_gtd[TO_TD_IND], &TOTD_VofS, e_gtd[TO_TD_IND], lambda, alpha_t, reward_val, gamma_t, gamma_tp1, x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[TO_TD_IND] += t2-t1;

                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            TD_lambda_sp(w_gtd[TD_IND], e_gtd[TD_IND], lambda, alpha_t, reward_val, gamma_t, gamma_tp1, x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[TD_IND] += t2-t1;

                            double rho_t = 1.0, rho_tm1 = 1.0, I = 1.0;
                            double to_etd_alpha = alpha_t;
                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            TO_ETD_sp(w_gtd[TO_ETD_IND], toetd_oldw, e_gtd[TO_ETD_IND], to_etd_alpha, &toetdF, &toetdD, I, lambda, rho_tm1,rho_t, reward_val, gamma_t, gamma_tp1,x_t,&x_tp1.vector, x_t_1s, &x_tp1_1s.vector);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[TO_ETD_IND] += t2-t1;

                            double beta = alpha_t/100;
                            double alpha2nd_atd = 1.0;
                            double atdxi = 0.01;
                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            atd_2ndorder_sp(w_gtd[ATD2nd_IND], b, e_gtd[ATD2nd_IND], &matu, &vecs, &matv, &matl, &matr, lambda, reward_val, gamma_t, gamma_tp1, x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector,  alpha2nd_atd, beta, atdxi, r, change_r, &updated_svd);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[ATD2nd_IND] += t2-t1;
                            
                            double tlstdxi = 0.01;
                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            T_lstd_sp(w_gtd[TLSTD_IND], b, e_gtd[TLSTD_IND], &matu, &vecs, &matv, &matl, &matr, lambda, reward_val, gamma_t, gamma_tp1,x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector, tlstdxi, r, change_r, &updated_svd);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[TLSTD_IND] += t2-t1;

                            double n0 = 100;
                            int m = 1;
                            double alpha_ilstd = alpha_t*(1+n0)/(n0+(double)episodeind);
                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            I_lstd_sp(ilstd_a, mu, w_gtd[ILSTD_IND], e_gtd[ILSTD_IND], lambda, alpha_ilstd, reward_val, gamma_t, gamma_tp1, x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector, m, change_r+1.0);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[ILSTD_IND] += t2-t1;

                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            LSTD_lambda_sherman(lstda, lstdb, w_gtd[LSTD_IND], e_gtd[LSTD_IND], lambda, reward_val, gamma_t, gamma_tp1, x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector, change_r);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[LSTD_IND] += t2-t1;

                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            PLSTD_lambda_avg(pa, plstdb, pw, pz, lambda, reward_val, gamma_t, gamma_tp1,x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector, skt, numofskt);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[PLSTD_IND] += t2-t1;

                            gsl_vector_memcpy (x_t, &x_tp1.vector);
                            gsl_vector_memcpy (x_t_1s, &x_tp1_1s.vector);
                            gamma_t = gamma_tp1;
                                
                              if (steps < totalsteps && steps % numerrorsample == 0) {
                                 for(xx=0; xx < nalgs;xx++)
                                 {
                                    double err = 0;
                                    gsl_vector_set_zero(mult);
                                    if (xx == PLSTD_IND) {
                                    gsl_vector *tempmult = gsl_vector_calloc(mult->size);
                                    for(int ss = 0; ss<numofskt; ss++){
                                       gsl_matrix_vector_product(tempmult, palls[ss], pw[ss]);
                                       gsl_blas_daxpy(1.0/(double)numofskt, tempmult, mult);    
                                    }
                                    gsl_vector_free(tempmult);
                                    }
                                    else
                                    gsl_blas_dgespmv(MDP.Allstates, allstates1indmat, w_gtd[xx], mult);
                                    err = compute_rmse(mult, MDP.Alltruevalues);
                                    mse_gtd[SixDto1DIndex(xx,as,bs,la,errsteps,rs,&sizes)] += err;
                                    var_mse_gtd[SixDto1DIndex(xx,as,bs,la,errsteps,rs,&sizes)] += err*err;

                                    if( steps % 20 == 0 )
                                      printf("the error of %s is: %f at step %d\n", names[xx], err, steps);
                                 }
                                 errsteps++;
                              }
                              change_r++;
                              steps++;
                            
                           } /* loop over steps in one episode */
                           episodeind++; 
                           //********reset matrices from reading outside files
                           gsl_matrix_free(onesind);
                           deallocate_mdp_t(&MDP);
                        } /* loop over episodes in one run */
                         
                        //********reset parameters for each algorithm at every run**********/
                        
                        gsl_matrix_free(matu);
                        gsl_vector_free(vecs);
                        gsl_matrix_free(matv);
                        gsl_matrix_free(matl);
                        gsl_matrix_free(matr);
                        gsl_vector_free(b);
                        
                        //********reset for TO-ETD algorithm
                        gsl_vector_free(toetd_oldw);
                        //*********reset for ilstd algorithm
                        gsl_matrix_free(ilstd_a);
                        gsl_vector_free(mu);
                        
                        gsl_matrix_free(lstda);
                        gsl_vector_free(lstdb);
                         
                        for(int i = 0; i< numofskt; i++){
                        gsl_matrix_free(palls[i]);
                        gsl_matrix_free(pa[i]);
                        gsl_vector_free(plstdb[i]);
                        gsl_matrix_free(skt[i]);
                        gsl_vector_free(pz[i]);
                        gsl_vector_free(pw[i]);
                        }
                        gsl_rng_free(rt.r);
                     }/* loop over runs */

                     /* after runs complete average data ------------- */
                     int xx, fiveind;
                     for(xx=0;xx<nalgs;xx++){
                        for (int ind = 0; ind < totalerrsteps; ind++) {
                           fiveind = SixDto1DIndex(xx,as,bs,la,ind,rs,&sizes);
                           mse_gtd[fiveind] = mse_gtd[fiveind] / ((double)numRuns);
                           var_mse_gtd[fiveind] = (var_mse_gtd[fiveind]) / ((double) numRuns) - mse_gtd[fiveind]*mse_gtd[fiveind];
                        }
                     }			
                  }/* loop over lambdas */
               }/* loop over alpha values */
            } /* loop over input rank values */
         } /* loop over beta values */
          
         free_vectors(w_gtd,nalgs);
         free_vectors(e_gtd,nalgs);
         gsl_vector_free(mult);
         gsl_vector_free(x_t);
         gsl_vector_free(x_t_1s);
         gsl_matrix_free(allstates1indmat);
      
            
         for(int zz=0;zz< totalSize;zz++)
         {	      
            mse_gtd_overall[zz] += mse_gtd[zz];
            var_mse_gtd_overall[zz] += sqrt(var_mse_gtd[zz])/sqrt((double)numRuns);
         }
            
         deallocate_truevalues_t(&MDP);

      } /* loop over MDPs */

      gsl_rng_free(rdfs);

      for(int zz=0;zz< totalSize;zz++)
      {	      
         mse_gtd_overall[zz] = mse_gtd_overall[zz]/(numMDP);
         var_mse_gtd_overall[zz] = var_mse_gtd_overall[zz]/(numMDP);
      }

      double num_to_scale = 1.0/(double)(totalsteps*numalpha*numlambda*numRuns);
      for (int xx = 0; xx < nalgs; xx++) {
         runningtime[xx] = runningtime[xx]*num_to_scale;
      }

      // store results for all prameter values for this MDP
      printf("DONE FINAL AVERAGING. NOW SAVE RESULTS TO FILE\n");
      printf("test");

      char * suffixes[3] = {"LC", "PramNames", "Var"};
      char filenames[3][5000];
      char prefix[1000];
      char runtimefilename[1000];
      memset(prefix, 0, sizeof(char)*1000);
      // argv[1] contains the index into the parameter setting
      if (argc > 1){
            sprintf(runtimefilename, "McarMDPresults/Mcarjobs/Mcar-Runtime-%d.txt", atoi(argv[1])); 
            sprintf(prefix, "McarMDPresults/Mcarjobs/Mcar-%d", atoi(argv[1]));
      }
      else{
            sprintf(runtimefilename, "%s", "McarMDPresults/Mcar-Runtime.txt");
            sprintf(prefix, "%s", "McarMDPresults/McarMDP");
      }

      FILE *runtimefile=fopen(runtimefilename, "w+");
      if(runtimefile != NULL){
         for(int xx=0;xx<nalgs;xx++){
            fprintf(runtimefile, "%s,%f\n",names[xx], runningtime[xx]);
         }
         fclose(runtimefile); 
      }

      for(int xx=0;xx<nalgs;xx++){
         printf("Running the saving results for loop!\n");
         
         for (int ss = 0; ss < 3; ss++) {
            memset(filenames[ss], 0, sizeof(char)*5000);
            sprintf(filenames[ss],"%s_feats_%d_steps_%d_%s_%s.txt",prefix,num_feature,totalsteps,names[xx],suffixes[ss]);
         }
            
         printf("Saving results to: %s\n",filenames[0]);

         FILE *matrix=fopen(filenames[0], "w+");
         FILE *matrix2=fopen(filenames[1], "w+");
         FILE *matrix3=fopen(filenames[2], "w+");
         
         if(matrix != NULL && matrix2 != NULL && matrix3 != NULL){
            int bs = 0;
            int rs = 0;
            for (bs = 0; bs < numbeta; bs++) {
               for (rs = 0; rs < numinputranks; rs++) {
                  for (int as =0; as < numalpha; as++) {
                     for (int la = 0; la < numlambda; la++)
                     {
                         //NOTE: record eta instead of beta
                        fprintf(matrix2, "%f %f %f %f %d\n",xis_range[as], alphas_range[as], etas_range[as], lambdas_range[la],inputranks_range[rs]);
                        for (int ind = 0; ind < totalerrsteps; ind++)
                        {
                           fprintf(matrix, "%f  ", mse_gtd_overall[SixDto1DIndex(xx,as,bs,la,ind,rs, &sizes)]);
                           fprintf(matrix3, "%f  ", var_mse_gtd_overall[SixDto1DIndex(xx,as,bs,la,ind, rs, &sizes)]);
                        }
                        fprintf(matrix, "\n");
                        fprintf(matrix3, "\n");
                     }
                  }
               }
            }
            fclose(matrix);
            fclose(matrix2);
            fclose(matrix3);
         }
         else
               printf("error opening file %s\n",filenames[0]);
      }
    
      free(M2_mse_gtd);
      free(mean_mse_gtd);
      free(var_mse_gtd);
      free(mse_gtd);
    
      printf("DONE SAVING. CLEANING UP.\n");
		
      return 1;
}

