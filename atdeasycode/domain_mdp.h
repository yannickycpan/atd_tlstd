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

/*
 * tabular features, sigma = 0.1. gamma=0.99
 * if alpha is NOT decayed on each step a good alpha range seems to be -7:0
 * if alpha IS decayed on each step a good alpha range seems to be -4:3
 *
 * RANDOM features, sigma = 0.1. gamma=0.9
 * if alpha is NOT decayed on each step a good alpha range seems to be -9:-2
 * if alpha is IS decayed on each step a good alpha range seems to be -5:2
 *r
 * RANDOM features, sigma = .5 gamma=0.9
 * if alpha is NOT decayed on each step a good alpha range seems to be -9:-2
 * if alpha is IS decayed on each step a good alpha range seems to be -5:2
 *
 * TABULAR features, sigma = .1 gamma=0.7
 * if alpha is NOT decayed on each step a good alpha range seems to be -7:0
 * if alpha IS decayed on each step a good alpha range seems to be -4:3
 */
/*
int mdp_trajectory[MAX_TRAJ_LEN];
double reward_trajectory[MAX_TRAJ_LEN];
double gamma_trajectory[MAX_TRAJ_LEN];
double rho_trajectory[MAX_TRAJ_LEN];
*/

struct domain_mdp_t {
    gsl_vector * rewards;
    gsl_matrix * PHI;
    gsl_matrix * Allstates;
    gsl_vector * Alltruevalues;
};

/* Allocation and creation functions */
void allocate_mdp_t(struct domain_mdp_t *MDP, const int numfs, const int biasunit, const int trajectory_len);

void deallocate_mdp_t(struct domain_mdp_t *MDP);

void deallocate_truevalues_t(struct domain_mdp_t *MDP);

void load_truevalues(struct domain_mdp_t *MDP, const int numfs, const int biasunit, const int numallstates, char * allstatesfilename, char * alltruevaluesfilename);

/* Returns 0 if success; otherwise, some return value indicating an error */
int getDomainMDP(struct domain_mdp_t *MDP, const int numfs, const int biasunit, const int trajectory_len, char * phifilename, char * rewardfilename);

#endif
