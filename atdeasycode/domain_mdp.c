#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "domain_mdp.h"
#include "utils.h"

void allocate_mdp_t(struct domain_mdp_t *MDP, const int numfs, const int biasunit, const int trajectory_len){
        //number of rewards is the length of trajectory
        MDP->rewards = gsl_vector_alloc(trajectory_len);
        //it is a matrix, numfs is number of features
        MDP->PHI = gsl_matrix_calloc(trajectory_len,numfs + biasunit);
}

void deallocate_mdp_t(struct domain_mdp_t *MDP){
        gsl_vector_free(MDP->rewards);
        gsl_matrix_free(MDP->PHI);
}

void deallocate_truevalues_t(struct domain_mdp_t *MDP){
        gsl_matrix_free(MDP->Allstates);
        gsl_vector_free(MDP->Alltruevalues);
}

void load_truevalues(struct domain_mdp_t *MDP, const int numfs, const int biasunit, const int numallstates, char * allstatesfilename, char * alltruevaluesfilename) {
        //another pointer to the matrix of all states
    //MDP->Allstates = gsl_matrix_alloc(numallstates,numfs + biasunit);
        //another pointer to the true value vector corresponding to all states
    //NOTE: ALLstates is not in USE
    MDP->Allstates = gsl_matrix_alloc(1,1);
    MDP->Alltruevalues = gsl_vector_alloc(numallstates);
    /*
    FILE * allstatesfile = fopen(allstatesfilename, "r");
    if (biasunit == 1) {
        gsl_matrix_view suball = gsl_matrix_submatrix (MDP->Allstates, 0, 0, (MDP->Allstates)->size1, (MDP->Allstates)->size2 - 1);
        gsl_matrix_fread(allstatesfile, &suball.matrix);
        gsl_vector_view unitcolinall = gsl_matrix_column (MDP->Allstates, (MDP->Allstates)->size2 - 1);
        gsl_vector_set_all(&unitcolinall.vector, 1.0);
    }
    else
        gsl_matrix_fread(allstatesfile, MDP->Allstates);
    //printf("read all states file \n");
    */
    FILE * alltruevaluesfile = fopen(alltruevaluesfilename, "r");
    gsl_vector_fread(alltruevaluesfile, MDP->Alltruevalues);
    //printf("the true values are: \n");
    //gsl_vector_print(MDP->Alltruevalues);
    
    //fclose(rewardfile);
    //fclose(allstatesfile);
    fclose(alltruevaluesfile); 
}

int getDomainMDP(struct domain_mdp_t * MDP, const int numfs, const int biasunit, const int trajectory_len, char * phifilename, char * rewardfilename) {
    
    allocate_mdp_t(MDP, numfs, biasunit, trajectory_len);
    /*
    FILE * phifile = fopen(phifilename, "r");
    if (biasunit == 1) {
        gsl_matrix_view subphi = gsl_matrix_submatrix (MDP->PHI, 0, 0, (MDP->PHI)->size1, (MDP->PHI)->size2 - 1);
        gsl_matrix_fread(phifile, &subphi.matrix);
        gsl_vector_view unitcol = gsl_matrix_column (MDP->PHI, (MDP->PHI)->size2 - 1);
        gsl_vector_set_all(&unitcol.vector, 1.0);
    }
    else gsl_matrix_fread(phifile, MDP->PHI);
    fclose(phifile);
     */
    //printf("finish allocating!\n");
    
    //FILE * rewardfile = fopen(rewardfilename, "r");
    //gsl_vector_fread(rewardfile, MDP->rewards);
    
    
    return 0;
}



