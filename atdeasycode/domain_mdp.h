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

struct domain_mdp_t {
    gsl_vector * rewards;
    gsl_matrix * PHI;
    gsl_matrix * Allstates;
    gsl_vector * Alltruevalues;
};

/* Allocation and creation functions */
void allocate_mdp_t(struct domain_mdp_t *MDP, const int numfs, const int biasunit, const int trajectory_len);

void deallocate_mdp_t(struct domain_mdp_t *MDP);

void deallocate_engy_mdp_t(struct domain_mdp_t *MDP);

void deallocate_truevalues_t(struct domain_mdp_t *MDP);

void load_truevalues(struct domain_mdp_t *MDP, const int numfs, const int biasunit, const int numallstates, char * allstatesfilename, char * alltruevaluesfilename);

/* Returns 0 if success; otherwise, some return value indicating an error, used to read in mountain car */
int getDomainMDP(struct domain_mdp_t *MDP, const int numfs, const int biasunit, const int trajectory_len, char * phifilename, char * rewardfilename);

int getEngyMDP(struct domain_mdp_t * MDP, const int numfs, const int biasunit, const int trajectory_len, const int numallstates, char * phifilename, char * rewardfilename, char * allstatesfilename, char * alltruevaluesfilename);

#endif
