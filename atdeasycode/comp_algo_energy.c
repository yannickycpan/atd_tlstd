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
        const int N = 5; /* number of states in MDP, NOT used */
        const int NS = 9999; /* number of steps per trajectory in an MDP */
        const int numRuns = 30; /* number of repeats */
        const int biasunit = 0;
        const int num_feature = 8192;
        const int trajectory_len = NS; //number of steps per trajectory
        const int numallstates = 2000;
        const int numerrorsample = 50;
        const int totalerrsteps = floor((double)NS/numerrorsample);
        char * trainfileprefix = "aaaifinalfs/";
        const int num_nz = 800;
    
        /* Algorithms options */
        double alphadiv = num_nz;
        int numalpha = 13;
        double alphas_range[13];
        for(int i = 0; i<numalpha; i++){
           alphas_range[i]=0.1*pow(2,i-7)/alphadiv;
        }
   
        int numxi = 13;
        double xis_range[13];
        for(int i = 0; i<numxi; i++){
          xis_range[i] = 0.1*pow(2, -11+i);
        }
 
        int numeta = 13;
        double etas_range[13];
        double etastep = 0;
        for(int i = 0; i<numeta; i++){
          etas_range[i] = pow(10, -4 + etastep);
          etastep += 0.75;
        }
        
        double lambdas_range[11] =  {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        int numlambda = sizeof(lambdas_range)/sizeof(double);
        printf("numlambda is %d\n",numlambda);

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
                        xis_range[0] = xis_range[i];
                        lambdas_range[0] = lambdas_range[j];
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
    
        int inputranks_range[1]= {30};
        // use beta range instead of rank for GTD
        double betas_range[1] = {pow(2, 1)};
        const int numbeta = sizeof(betas_range)/sizeof(double);
        const int numinputranks = sizeof(inputranks_range)/sizeof(int);
    
        //const int numbeta = 1; // since on-policy, ignore beta parameter
        const int decay_alpha = 0;
        printf("numinputrank is %d\n",numinputranks);
        printf("numruns is %d\n",numRuns);
        /* Algorithms */
        const int nalgs = 7;
      const char *names[nalgs];
      const int TD_IND = 0;
      const int TO_TD_IND = 1;
      const int TO_ETD_IND = 2;
      const int ATD2nd_IND = 3;
      const int TLSTD_IND = 4;
      const int FLSTD_SA_IND = 5;
      const int PLSTD_IND = 6;

      names[TD_IND] = "TD";
      names[TO_TD_IND] = "TO-TD";
      names[TO_ETD_IND] = "TO-ETD";
      names[FLSTD_SA_IND] = "FLSTD-SA";
      names[ATD2nd_IND] = "ATD2nd";
      names[TLSTD_IND] = "TLSTD";
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
        //set up file names which will be used by MDP
        char * allstatesfile = concat(trainfileprefix,"codedevlstates2000");
        char * truevaluefile = concat(trainfileprefix,"evlvalues2000");
        char * allstatesfile1indexes = concat(trainfileprefix, "codedevlstates1indexes2000");
        FILE *allstates1indexes = fopen(allstatesfile1indexes, "r");
        gsl_matrix * allstates1indmat = gsl_matrix_alloc(numallstates, num_nz + biasunit);
        if (biasunit == 1) {
            gsl_matrix_view suball1indmat = gsl_matrix_submatrix(allstates1indmat, 0, 0, allstates1indmat->size1, allstates1indmat->size2 - 1);
            gsl_matrix_fread(allstates1indexes, &suball1indmat.matrix);
            gsl_vector_view unitcolall = gsl_matrix_column(allstates1indmat, allstates1indmat->size2 - 1);
            gsl_vector_set_all(&unitcolall.vector, num_feature);
        }
        else gsl_matrix_fread(allstates1indexes, allstates1indmat);
        fclose(allstates1indexes);
        //printf("current here -----------\n");
    
        int k;
        for (k = 0; k < numMDP; k++) {

      
            /*--------------------- store best performance for each MDP */
            memset (mse_gtd, 0, totalSize*sizeof(double));
            memset (var_mse_gtd, 0, totalSize*sizeof(double));
            memset (mean_mse_gtd, 0, totalSize*sizeof(double));
            memset (M2_mse_gtd, 0, totalSize*sizeof(double));
            /*--------------------- store best performance for each MDP */
            
            double TOTD_VofS;

            //mult will store estimated values for all states
            gsl_vector * mult = gsl_vector_alloc(numallstates);
            //note: x_t is not used with sparse computation
            gsl_vector * x_t = gsl_vector_alloc(num_feature + biasunit);
            gsl_vector * x_t_1s = gsl_vector_alloc(num_nz + biasunit);

            gsl_vector * w_gtd[nalgs];
            alloc_vectors(w_gtd, nalgs, num_feature + biasunit);
            gsl_vector * e_gtd[nalgs];
            alloc_vectors(e_gtd, nalgs, num_feature + biasunit);
            
                
            int as;
            int bs;
            int rs; // input rank to iterate
            for (bs = 0; bs < numbeta; bs++){
            //beta is not used
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
                    printf("Running (alpha, lambda) = (%g, %g)\n", alpha, lambda);
                    int z;
                    for (z = 0; z < numRuns; z++)
                    {   
                        int LEN = CHAR_BIT * sizeof(int)/3 + 2;
                        char index[LEN];
                        snprintf(index, LEN, "%d", z + 1);
                        char * phifilename = concat(trainfileprefix, concat("codedphi",index));
                        char * rewardfilename = concat(trainfileprefix,concat("rewards",index));
                        printf("filename is %s\n",rewardfilename);
                        getEngyMDP(&MDP, num_feature, biasunit, trajectory_len, numallstates, phifilename, rewardfilename, allstatesfile, truevaluefile);
                        //printf("got the mdp!\n");
                        
                        //read sparse matrices indexes
                        char * onesindexesfilename = concat(trainfileprefix, concat("codedphi1indexes",index));
                        FILE * onesindexesfile = fopen(onesindexesfilename,"r");
                        gsl_matrix * onesind = gsl_matrix_alloc(trajectory_len, num_nz + biasunit);
                        if (biasunit == 1) {
                            gsl_matrix_view sub1indmat = gsl_matrix_submatrix(onesind, 0, 0, onesind->size1, onesind->size2 - 1);
                            gsl_matrix_fread(onesindexesfile, &sub1indmat.matrix);
                            gsl_vector_view unitcol = gsl_matrix_column(onesind, onesind->size2 - 1);
                            gsl_vector_set_all(&unitcol.vector, num_feature);
                        }
                        else gsl_matrix_fread(onesindexesfile, onesind);
                        fclose(onesindexesfile);
                        
                        // Initialize state, gamma, etc.
                        gsl_vector_view x_t_1s_view = gsl_matrix_row(onesind, 0);
                        gsl_vector_memcpy(x_t_1s, &x_t_1s_view.vector);
 
                        double reward_val = gsl_vector_get(MDP.rewards, 0);
                        
                        int xx;
                        for(xx=0;xx<nalgs;xx++){
                            gsl_vector_set_zero(w_gtd[xx]);
                            gsl_vector_set_zero(e_gtd[xx]);
                        }

                        /* For True-online TD algorithm as baseline */
                        TOTD_VofS=0;
                        // For TO-ETD algorithm, toetd_oldw is not used
                        double toetdF = 0, toetdD = 0;
                        gsl_vector *toetd_oldw = gsl_vector_calloc(1);

		                // For tLSTD algorithm
			            int r = inputrank;
                        int change_r = 0;
                       
                        // For random shuffle TD
                        int shuffle[30-1];
                        for(int i = 0; i<r-1; i++)shuffle[i] = i;
                        gsl_rng * rd = gsl_rng_alloc (gsl_rng_taus);
 
                        //tlstd, atd, nonsampled use the same A matrix and b vector
                        gsl_matrix * matu = gsl_matrix_calloc(x_t->size,1);
                        gsl_vector * vecs = gsl_vector_calloc(1);
			            gsl_matrix * matv = gsl_matrix_calloc(x_t->size,1);
			            gsl_matrix * matl = gsl_matrix_alloc(1,1);
                        gsl_matrix * matr = gsl_matrix_alloc(1,1);
                        gsl_matrix_set_identity(matl);
                        gsl_matrix_set_identity(matr);
                        gsl_vector * b = gsl_vector_calloc(x_t->size);
                        
                        // For random projection
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
                        
                        // Indicator variable for whether svd is updated or not
                        int updated_svd = 0;
                        
                        double gamma_t = 0, gamma_tp1 = 0.99;
                        // compute error before learning
                        int i = 0, errstep = 0;
                        for(xx=0; xx<nalgs;xx++){
                            gsl_blas_dgespmv(MDP.Allstates, allstates1indmat, w_gtd[xx], mult);
                            double err = compute_pme(mult, MDP.Alltruevalues);
                            mse_gtd[SixDto1DIndex(xx,as,bs,la,errstep,rs,&sizes)] += err;
                            var_mse_gtd[SixDto1DIndex(xx,as,bs,la,errstep,rs,&sizes)] += err*err;
                        }
                        errstep++;
                        for (i = 1; i < NS; i++) {
                            updated_svd = 0;
                           
                            //NOTE: x_t and x_tp1 are not used with sparse computation 
                            gsl_vector_view x_tp1 = gsl_vector_subvector(x_t, 0, x_t->size);
                            gsl_vector_view x_tp1_1s = gsl_matrix_row(onesind, i);
                           
                            double alpha_t;
                            if (decay_alpha == 1)
                                    alpha_t = alpha*1000.0 / (1000 + pow(i, (1.0 / 3.0)));
                            else
                                    alpha_t = alpha;
                           
                            gettimeofday(&tim, NULL);
                           t1 = tim.tv_sec+(tim.tv_usec/1000000.0); 
                           if(i >= r && (change_r+1)%r == 0){
                              gsl_ran_shuffle(rd, shuffle, plstdr, sizeof(int));
                              for(int ii = 0; ii < r-1; ii++){
                              int tupleindex = (i-r) + shuffle[ii];  
                              double gamma_tp1_td = gamma_tp1;
                              gsl_vector_view x_td_1s = gsl_matrix_row(onesind, tupleindex);
                              gsl_vector_view x_tp_td_1s = gsl_matrix_row(onesind, tupleindex+1);
                              double reward_td =  gsl_vector_get(MDP.rewards, tupleindex);
                              TD_lambda_sp(w_gtd[FLSTD_SA_IND], e_gtd[FLSTD_SA_IND], 0, alpha_t, reward_td, gamma_t, gamma_tp1_td, x_t, &x_tp1.vector, &x_td_1s.vector, &x_tp_td_1s.vector);
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
                            

                            double beta = 0.000001;
                            double alpha2nd_atd = 1.0;
                            double atdxi = xi;
                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            atd_2ndorder_sp(w_gtd[ATD2nd_IND], b, e_gtd[ATD2nd_IND], &matu, &vecs, &matv, &matl, &matr, lambda, reward_val, gamma_t, gamma_tp1, x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector, alpha2nd_atd, beta, atdxi, r, change_r, &updated_svd);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[ATD2nd_IND] += t2-t1;
                            
                            double tlstdxi = xi;
                            gettimeofday(&tim, NULL);
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            T_lstd_sp(w_gtd[TLSTD_IND], b, e_gtd[TLSTD_IND], &matu, &vecs, &matv, &matl, &matr, lambda, reward_val, gamma_t, gamma_tp1,x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector, tlstdxi, r, change_r, &updated_svd);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[TLSTD_IND] += t2-t1;
                             
                            t1 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            PLSTD_lambda_avg(pa, plstdb, pw, pz, lambda, reward_val, gamma_t, gamma_tp1,x_t, &x_tp1.vector, x_t_1s, &x_tp1_1s.vector, skt, numofskt);
                            gettimeofday(&tim, NULL);
                            t2 = tim.tv_sec+(tim.tv_usec/1000000.0);
                            runningtime[PLSTD_IND] += t2-t1;
  
                            change_r++;
							
                            /* update error for each alg every 50 step */
                            if (i % numerrorsample == 0 && errstep < totalerrsteps) {
                            for(xx=0;xx<nalgs;xx++)
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
                                err = compute_pme(mult, MDP.Alltruevalues);
                                mse_gtd[SixDto1DIndex(xx,as,bs,la,errstep,rs,&sizes)] += err;
                                var_mse_gtd[SixDto1DIndex(xx,as,bs,la,errstep,rs,&sizes)] += err*err;
                                //if(i%10==0)
                                //   printf("the error of %d is: %f at step %d \n", xx, err, i);
                            }
                            errstep++;
                            } //end if for computing error
                            
                            gsl_vector_memcpy (x_t_1s, &x_tp1_1s.vector);
                            gamma_t = gamma_tp1;
                            reward_val = gsl_vector_get(MDP.rewards, i);
                        } /* loop over steps in trajectory */
		                
                        //********reset parameters for tlstd at every run
                        
                        gsl_matrix_free(matu);
                        gsl_vector_free(vecs);
                        gsl_matrix_free(matv);
                        gsl_matrix_free(matl);
                        gsl_matrix_free(matr);
                        gsl_vector_free(b);

                        for(int i = 0; i< numofskt; i++){
                        gsl_matrix_free(palls[i]);
                        gsl_matrix_free(pa[i]);
                        gsl_vector_free(plstdb[i]);
                        gsl_matrix_free(skt[i]);
                        gsl_vector_free(pz[i]);
                        gsl_vector_free(pw[i]);
                        }
                        
                        gsl_rng_free(rt.r);

                        //********reset matrices from reading outside files
                        gsl_matrix_free(onesind);
                        deallocate_engy_mdp_t(&MDP);
                        
                        gsl_vector_free(toetd_oldw);

                    }/* loop over runs */

                    /* after runs complete average data ------------- */
                    int xx, fiveind;
                    for(xx=0;xx<nalgs;xx++){
                        for (int ind = 0; ind < totalerrsteps; ind++) {
                            fiveind = SixDto1DIndex(xx,as,bs,la,ind,rs,&sizes);
                            mse_gtd[fiveind] = mse_gtd[fiveind] / ((double)numRuns);
                            var_mse_gtd[fiveind] = (var_mse_gtd[fiveind]) / ((double)numRuns) - mse_gtd[fiveind]*mse_gtd[fiveind];
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
                var_mse_gtd_overall[zz] += var_mse_gtd[zz];
            }
            
            double num_to_scale = 1.0/(double)(NS*numalpha*numlambda*numRuns);
            for (int xx = 0; xx < nalgs; xx++) {
                runningtime[xx] = runningtime[xx]*num_to_scale;
            }

            deallocate_mdp_t(&MDP);
        } /* loop over MDPs */


        for(int zz=0;zz< totalSize;zz++)
        {	      
            mse_gtd_overall[zz] = mse_gtd_overall[zz]/(numMDP);
            var_mse_gtd_overall[zz] = var_mse_gtd_overall[zz]/(numMDP);
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
        sprintf(runtimefilename, "EnergyMDPresults/Engyjobs/Engy-Runtime-%d.txt", atoi(argv[1]));
        sprintf(prefix, "EnergyMDPresults/Engyjobs/Engy-%d", atoi(argv[1]));
    }
    else{
        sprintf(runtimefilename, "%s", "EnergyMDPresults/Engy-Runtime.txt");
        sprintf(prefix, "%s", "EnergyMDPresults/EngyMDP");
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
            sprintf(filenames[ss],"%s_feats_%d_steps_%d_%s_%s.txt",prefix,num_feature,NS,names[xx],suffixes[ss]);
        }
        
        printf("Saving results to: %s\n",filenames[0]);

            FILE *matrix=fopen(filenames[0], "w+");
            FILE *matrix2=fopen(filenames[1], "w+");
            FILE *matrix3=fopen(filenames[2], "w+");
      
            if(matrix != NULL &&matrix2 != NULL && matrix3 != NULL){
                int bs = 0;
                int rs = 0;
                for (bs = 0; bs < numbeta; bs++) {
                for (rs = 0; rs < numinputranks; rs++) {
                for (int as =0; as < numalpha; as++) {
                    for (int la = 0; la < numlambda; la++)
                    {
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
            }
            else
                    printf("error opening file %s\n",names[0]);
        }

        free(M2_mse_gtd);
        free(mean_mse_gtd);
        free(var_mse_gtd);
        free(mse_gtd);
    
        printf("DONE SAVING. CLEANING UP.\n");
		
        return 1;
}

