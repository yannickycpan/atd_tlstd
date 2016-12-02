#ifndef _ALGORITHMS_H
#define _ALGORITHMS_H

#define MAX_AGENT_PRAMS 50

#include "utils.h"

int TD_lambda_sp(gsl_vector * w, gsl_vector * e, const double lambda_t, const double alpha_t, const double reward, const double gamma_t, double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s);

int TO_TD_lambda_sp(gsl_vector * w,double * VofS, gsl_vector * e, const double lambda_t, const double alpha_t, const double reward, const double gamma_t, const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s);

int TO_ETD_sp(gsl_vector * w,  gsl_vector * wold, gsl_vector * e, double alpha, double * F, double * D, double I, double lambda_t, double rho_tm1, double rho_t, double reward, double gamma_t, double gamma_tp1,gsl_vector * x_t, gsl_vector* x_tp1,const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s);

int T_lstd(gsl_vector * w, gsl_vector * b, gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const double xi, int r, int change_r,  int * updated_svd);

int atd_2ndorder_sp(gsl_vector * w, gsl_vector* b, gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const double alpha_t, double beta_t, const double xi, int r, int change_r, int * updated_svd);

int T_lstd_sp(gsl_vector * w, gsl_vector * b, gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const double xi, int r, int change_r, int * updated_svd);

int T_lstd_batch_sp(gsl_vector * w, gsl_vector * b, gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix * matz, gsl_matrix * matd, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const double xi, int r, int k, int change_r, int * updated_svd);

int Update_svd(gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const gsl_vector * z, const gsl_vector * d, int r, int count);

int Update_batch_svd(gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix * matz, gsl_matrix * matd, int r, int k, int count);

int LSTD_lambda_sherman(gsl_matrix * a, gsl_vector *b, gsl_vector * w, gsl_vector * z, const double lambda_t, const double reward, const double gamma_t, const double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const int change_r);

int PLSTD_sherman_sp(gsl_matrix * a, gsl_vector *b, gsl_vector * w, gsl_vector * z, const double lambda_t, const double reward, const double gamma_t, const double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const gsl_matrix * projmat);

int PLSTD_lambda_avg(gsl_matrix * a[], gsl_vector *b[], gsl_vector * w[], gsl_vector * z[], const double lambda_t, const double reward, const double gamma_t, const double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, gsl_matrix * projmat[], const int numofskt);

int I_lstd_sp(gsl_matrix * mata, gsl_vector *mu, gsl_vector * w, gsl_vector * z, const double lambda_t, const double alpha_t, const double reward, const double gamma_t, const double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const int m, const int numsteps);

/* different components of a value learning algorithm */
int accumulate_trace(gsl_vector * z, const double lambda_t, const double gamma_t, const gsl_vector * x_t, const gsl_vector * x_t_1s);

gsl_vector * compute_dvec(const double gamma_tp1, const gsl_vector * x_t, const gsl_vector * x_t_1s, const gsl_vector* x_tp1, const gsl_vector* x_tp1_1s);

void update_bvec(gsl_vector * b,gsl_vector * z,const double reward,int change_r);

int update_svd_1ststep(const gsl_vector *z, const gsl_vector *d, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv);

int updated_svd_algorithms_sp(gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const double xi, int r, int change_r, int * updated_svd);
#endif
