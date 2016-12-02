#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "algorithms.h"

/* sparse version of TD_lambda */
int TD_lambda_sp(gsl_vector * w, gsl_vector * e, const double lambda_t, const double alpha_t, const double reward, const double gamma_t, double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s){

    double old_v = 0;
    gsl_blas_spddot(w, x_t, x_t_1s, &old_v);
    double new_v = 0;
    gsl_blas_spddot (w, x_tp1, x_tp1_1s, &new_v);
    
    double delta = reward + gamma_tp1*new_v - old_v;
    gsl_vector_scale(e,gamma_t*lambda_t);
    for (int i = 0; i < x_t_1s->size; i++) {
        gsl_vector_set(e, (int)gsl_vector_get(x_t_1s, i), 1.0);
    }
    gsl_blas_daxpy (alpha_t*delta, e, w);
    return 0;
}

/* sparse version of TO_TD_lambda */
/* Store current value in VofS, because need that value for the next step */
int TO_TD_lambda_sp(gsl_vector * w,double * VofS, gsl_vector * e, const double lambda_t, const double alpha_t, const double reward, const double gamma_t, const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s){
    
    double new_v = 0;
    gsl_blas_spddot(w, x_tp1, x_tp1_1s, &new_v);
    double delta = reward + gamma_tp1*new_v - *VofS;
    
    double dot;
    gsl_blas_spddot(e, x_t, x_t_1s, &dot);
    double a = alpha_t*(1.0 - gamma_t*lambda_t*dot);
    gsl_vector_scale(e,gamma_t*lambda_t);
    gsl_blas_spdaxpy(a, x_t, x_t_1s, e);
    
    dot = 0;
    gsl_blas_spddot (w, x_t, x_t_1s, &dot);
    a = alpha_t*(*VofS-dot);
    gsl_blas_daxpy (delta, e, w);
    gsl_blas_spdaxpy (a, x_t, x_t_1s, w);
    *VofS = new_v;
    
    return 0;
}

int TO_ETD_sp(gsl_vector * w,  gsl_vector * wold, gsl_vector * e, double alpha, double * F, double * D, double I, double lambda_t, double rho_tm1, double rho_t, double reward, double gamma_t, double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1,const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s)
{
    double old_v = 0;
    gsl_blas_spddot (w, x_t, x_t_1s, &old_v);
    double new_v = 0;
    gsl_blas_spddot (w, x_tp1, x_tp1_1s, &new_v);
    
    double delta = reward + gamma_tp1*new_v - old_v;
    *F = *F*rho_tm1*gamma_t + I;
    double M = lambda_t*I + (1-lambda_t)*(*F);
    double phie = 0;
    gsl_blas_spddot (e, x_t, x_t_1s, &phie);
    
    double S = rho_t*alpha*M * (1 - rho_t*gamma_t*lambda_t*phie);
    gsl_vector_scale(e,rho_t*gamma_t*lambda_t);
    gsl_blas_spdaxpy (S, x_t, x_t_1s, e);
    
    /* Rich's approach */
    gsl_vector * dif = gsl_vector_calloc(w->size);
    //gsl_vector_memcpy(dif,x_t);
    gsl_spvector_scale(dif, x_t_1s, -alpha*M*rho_t);
    gsl_vector_add(dif,e);
    gsl_vector_scale(dif,*D);
    gsl_blas_daxpy (delta, e, dif);
    gsl_vector_add(w,dif);
    
    gsl_blas_spddot (dif, x_tp1, x_tp1_1s, D);
    gsl_vector_free(dif);
    
    return 0;
}

// The bvec is only used if epsilon > 0
int atd_2ndorder_sp(gsl_vector * w, gsl_vector* b, gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const double alpha_t, double beta_t, const double xi, int r, int change_r, int * updated_svd){
    
    //update z vector
    gsl_vector_scale(z, gamma_t*lambda_t);
    for (int i = 0; i < x_t_1s->size; i++) {
        gsl_vector_set(z, (int)gsl_vector_get(x_t_1s, i), 1.0);
    }
    
    //compute delta
    double old_v = 0;
    gsl_blas_spddot(w, x_t, x_t_1s, &old_v);
    double new_v = 0;
    gsl_blas_spddot (w, x_tp1, x_tp1_1s, &new_v);
    double delta = reward + gamma_tp1*new_v - old_v;

    if(*updated_svd == 0) {
        update_bvec(b, z, reward, change_r);
        updated_svd_algorithms_sp(z, matu, vecs, matv, matl, matr, lambda_t, reward, gamma_t,gamma_tp1,  x_t, x_tp1, x_t_1s, x_tp1_1s,  xi,  r, change_r, updated_svd);
    }
    
    gsl_vector * inverse_sigma = gsl_vector_calloc ((*vecs)->size);
    for (int i = 0; i < (int)(*vecs)->size; i++){
        double val = 0;
        if(gsl_vector_get(*vecs, i) > xi*gsl_vector_get(*vecs,0))
            val = 1.0/(gsl_vector_get(*vecs, i));
        gsl_vector_set(inverse_sigma, i, val);
    }

    //do matrix computation, start from the rightmost side to get a vector first
    gsl_vector *temp = gsl_vector_calloc((*matu)->size1);
    gsl_blas_daxpy (delta, z, temp);
    
    gsl_vector * tempv1 = gsl_vector_alloc((*matu)->size2);
    gsl_vector * tempv2 = gsl_vector_alloc((*matl)->size2);
    gsl_blas_dgemv (CblasTrans, 1.0, *matu, temp, 0.0, tempv1);
    gsl_blas_dgemv (CblasTrans, 1.0, *matl, tempv1, 0.0, tempv2);
    gsl_vector_mul(tempv2, inverse_sigma);
    gsl_blas_dgemv (CblasNoTrans, 1.0, *matr, tempv2, 0.0, tempv1);
    gsl_blas_dgemv (CblasNoTrans, alpha_t/((double)change_r + 1.0), *matv, tempv1, 1.0, w);
    
    //add the regularization part
    gsl_blas_daxpy(beta_t, temp, w);

    gsl_vector_free(temp);
    gsl_vector_free(tempv1);
    gsl_vector_free(tempv2);
    gsl_vector_free(inverse_sigma);
    
    return 1;
}

int T_lstd_sp(gsl_vector * w, gsl_vector * b, gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const double xi, int r, int change_r, int * updated_svd){
    
    //update z vector
    gsl_vector_scale(z, gamma_t*lambda_t);
    for (int i = 0; i < x_t_1s->size; i++) {
        gsl_vector_set(z, (int)gsl_vector_get(x_t_1s, i), 1.0);
    }
    if (*updated_svd == 0) {
        update_bvec(b,z, reward,change_r);
        updated_svd_algorithms_sp(z, matu, vecs, matv, matl, matr, lambda_t, reward, gamma_t,gamma_tp1,  x_t, x_tp1, x_t_1s, x_tp1_1s,  xi,  r,  change_r, updated_svd);
    }
    
    //Last part: update w, first compute the inverse of diagonal matrix Sigma
    gsl_vector * inverse_sigma = gsl_vector_calloc ((*vecs)->size);
    int rank = 0;
    for (int i = 0; i < (int)(*vecs)->size; i++){
        double val = 0;
        if(gsl_vector_get(*vecs, i) > xi){
            val = 1.0/gsl_vector_get(*vecs, i);
            rank++;
        }
        gsl_vector_set(inverse_sigma, i, val);
    }
    
    
    gsl_vector * tempv1 = gsl_vector_calloc((*matu)->size2);
    gsl_blas_dgemv (CblasTrans, 1.0, *matu, b, 0.0, tempv1);
    gsl_vector * tempv2 = gsl_vector_calloc((*matl)->size2);
    gsl_blas_dgemv (CblasTrans, 1.0, *matl, tempv1, 0.0, tempv2);
    gsl_vector_mul(tempv2, inverse_sigma);
    gsl_blas_dgemv (CblasNoTrans, 1.0, *matr, tempv2, 0.0, tempv1);
    gsl_blas_dgemv (CblasNoTrans, 1.0, *matv, tempv1, 0.0, w);
    
    gsl_vector_free(tempv1);
    gsl_vector_free(tempv2);
    gsl_vector_free(inverse_sigma);
    return 0;
}

int T_lstd_batch_sp(gsl_vector * w, gsl_vector * b, gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix * matz, gsl_matrix * matd, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const double xi, int r, int k, int change_r, int * updated_svd){
    
    //update z vector
    gsl_vector_scale(z, gamma_t*lambda_t);
    for (int i = 0; i < x_t_1s->size; i++) {
        gsl_vector_set(z, (int)gsl_vector_get(x_t_1s, i), 1.0);
    }
    
    //compute d vector to D mat
    gsl_vector_view row_matd = gsl_matrix_column(matd, change_r % k);
    gsl_vector_set_zero(&row_matd.vector);
    gsl_blas_spdaxpy(-gamma_tp1, x_tp1, x_tp1_1s, &row_matd.vector);
    gsl_spvector_add(&row_matd.vector, x_t, x_t_1s);
    //compute z vector to Z mat
    gsl_vector_view row_matz = gsl_matrix_column(matz, change_r % k);
    gsl_vector_memcpy(&row_matz.vector, z);
    
    double beta = 1.0/(double)(change_r + 1.0);
    gsl_vector_scale(b, 1.0 - beta);
    gsl_blas_daxpy(reward*beta, z, b);
    //gsl_blas_daxpy(reward, z, b);
    if ((change_r+1) % k == 0 && (*updated_svd == 0)) {
        beta = (double)change_r/(change_r+k);
        gsl_vector_scale(*vecs, beta);
        Update_batch_svd(matu, vecs, matv, matz, matd, r, k, change_r);
        *updated_svd = 1;
    }
    else if((change_r+1) % k != 0) return 0;
    
    //Last part: update w, first compute the inverse of diagonal matrix Sigma
    gsl_vector * inverse_sigma = gsl_vector_calloc ((*vecs)->size);
    int rank = 0;
    for (int i = 0; i < (int)(*vecs)->size; i++){
        double val = 0;
        //printf("the max sigma is %f\n",gsl_vector_get(*vecs,0));
        if(gsl_vector_get(*vecs, i) > xi*gsl_vector_get(*vecs, 0)){
            val = 1.0/gsl_vector_get(*vecs, i);
            rank++;
        }
        gsl_vector_set(inverse_sigma, i, val);
    }
    
    gsl_vector * tempv1 = gsl_vector_calloc((*matu)->size2);
    gsl_blas_dgemv (CblasTrans, 1.0, *matu, b, 0.0, tempv1);
    gsl_vector_mul(tempv1, inverse_sigma);
    gsl_blas_dgemv (CblasNoTrans, 1.0, *matv, tempv1, 0.0, w);
    
    gsl_vector_free(tempv1);
    gsl_vector_free(inverse_sigma);
    return 0;
}

int LSTD_lambda_sherman(gsl_matrix * a, gsl_vector *b, gsl_vector * w, gsl_vector * z, const double lambda_t, const double reward, const double gamma_t, const double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const int change_r){
    
    gsl_vector_scale(z, gamma_t*lambda_t);
    for(int i = 0; i<x_t_1s->size; i++)
        gsl_vector_set(z, (int)gsl_vector_get(x_t_1s, i), 1);
    //compute vector d
    gsl_vector * d = gsl_vector_calloc(x_t->size);
    gsl_blas_daxpy(-gamma_tp1, x_tp1, d);
    gsl_vector_add(d, x_t);
    
    //update vector b
    gsl_blas_daxpy(reward, z, b);
    
    //update A matrix
    gsl_vector *work = gsl_vector_alloc(x_t->size);
    gsl_blas_dgemv (CblasNoTrans, 1.0, a, z, 0, work);
    double denominator = 0;
    gsl_blas_ddot(d, work, &denominator);
    denominator+=1;
    gsl_vector *work1 = gsl_vector_alloc(x_t->size);
    gsl_blas_dgemv (CblasTrans, 1.0, a, d, 0, work1);
    gsl_matrix *delta_a = gsl_matrix_alloc(x_t->size, x_t->size);
    
    gsl_outer_product(delta_a, work, work1);
    
    gsl_matrix_scale(delta_a, -1.0/denominator);
    gsl_matrix_add(a, delta_a);
    
    gsl_blas_dgemv (CblasNoTrans, 1.0, a, b, 0, w);
    gsl_vector_free(work);
    gsl_vector_free(work1);
    
    gsl_vector_free(d);
    gsl_matrix_free(delta_a);
    return 0;
}

int PLSTD_sherman_sp(gsl_matrix * a, gsl_vector *b, gsl_vector * w, gsl_vector * z, const double lambda_t, const double reward, const double gamma_t, const double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const gsl_matrix * projmat){
    
    gsl_vector * px_t = gsl_vector_alloc(projmat->size2);
    gsl_vector * px_tp1 = gsl_vector_alloc(projmat->size2);
    gsl_blas_dgespvm(projmat, x_t, x_t_1s, px_t);
    gsl_blas_dgespvm(projmat, x_tp1, x_tp1_1s, px_tp1);
    
    gsl_vector_scale(z, gamma_t*lambda_t);
    gsl_vector_add(z, px_t);
    
    gsl_vector * d = gsl_vector_calloc(px_t->size);
    gsl_blas_daxpy(-gamma_tp1, px_tp1, d);
    gsl_vector_add(d, px_t);
    
    gsl_blas_daxpy(reward, z, b);
    
    gsl_vector *work = gsl_vector_alloc(px_t->size);
    gsl_blas_dgemv (CblasNoTrans, 1.0, a, z, 0, work);
    double denominator = 0;
    gsl_blas_ddot(d, work, &denominator);
    denominator+=1;
    gsl_vector *work1 = gsl_vector_alloc(px_t->size);
    gsl_blas_dgemv (CblasTrans, 1.0, a, d, 0, work1);
    gsl_matrix *delta_a = gsl_matrix_alloc(px_t->size, px_t->size);
    
    gsl_outer_product(delta_a, work, work1);
    
    gsl_matrix_scale(delta_a, -1.0/denominator);
    gsl_matrix_add(a, delta_a);
    
    gsl_blas_dgemv (CblasNoTrans, 1.0, a, b, 0, w);
    gsl_vector_free(work);
    gsl_vector_free(work1);
    
    gsl_vector_free(d);
    gsl_matrix_free(delta_a);
    
    gsl_vector_free(px_t);
    gsl_vector_free(px_tp1);
    
    return 0;
}

int PLSTD_lambda_avg(gsl_matrix * a[], gsl_vector *b[], gsl_vector * w[], gsl_vector * z[], const double lambda_t, const double reward, const double gamma_t, const double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, gsl_matrix * projmat[], const int numofskt){
    
    for (int i = 0; i<numofskt; i++) {
        PLSTD_sherman_sp(a[i], b[i], w[i], z[i], lambda_t, reward, gamma_t, gamma_tp1, x_t, x_tp1, x_t_1s, x_tp1_1s, projmat[i]);
    }
    
    return 0;
}


int I_lstd_sp(gsl_matrix * mata, gsl_vector *mu, gsl_vector * w, gsl_vector * z, const double lambda_t, const double alpha_t, const double reward, const double gamma_t, const double gamma_tp1, const gsl_vector * x_t, const gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const int m, const int numsteps){
    
    //update z, eligibility trace
    gsl_vector_scale(z, gamma_t*lambda_t);
    //gsl_spvector_add(z, x_t, x_t_1s);
    for (int i = 0; i < x_t_1s->size; i++) {
        gsl_vector_set(z, (int)gsl_vector_get(x_t_1s, i), 1.0);
    }
    //compute delta_b
    gsl_vector* delta_b = gsl_vector_calloc(z->size);
    gsl_blas_daxpy(reward, z, delta_b);
    
    //compute vector d, used to compute delta_a
    gsl_vector * d = gsl_vector_calloc(x_t->size);
    gsl_blas_spdaxpy(-gamma_tp1, x_tp1, x_tp1_1s, d);
    gsl_spvector_add(d, x_t, x_t_1s);
    
    //compute outer product, z*d^T and directly update mata
    //nzs stores non-zero indexes in vector d
    gsl_vector * nzs = gsl_spvector_get_nz2vecs(x_t_1s, x_tp1_1s);
    for (int i = 0; i < nzs->size; i++) {
        int row_ind = (int)gsl_vector_get(nzs, i);
        for (int j = 0; j < nzs->size; j++) {
            int col_ind = (int)gsl_vector_get(nzs, j);
            double changenum = gsl_vector_get(z,row_ind)*gsl_vector_get(d, col_ind);
            double originalnum = gsl_matrix_get(mata, row_ind, col_ind);
            gsl_matrix_set(mata, row_ind, col_ind, originalnum + changenum);
        }
    }
    
    //update vector mu, compute <(phi - gamma*phi'), w> first, then scale z
    double dotphiw = 0;
    gsl_blas_ddot(w, d, &dotphiw);
    gsl_vector *delta_aw = gsl_vector_calloc(mu->size);
    gsl_blas_daxpy(dotphiw, z, delta_aw);
    gsl_vector_add(mu, delta_b);
    gsl_vector_sub(mu, delta_aw);
    
    //partially update theta
    gsl_vector * acol = gsl_vector_alloc(mata->size1);
    for(int i = 0; i<m; i++){
        int j = gsl_blas_idamax (mu);
        gsl_vector_set(w, j, gsl_vector_get(w, j) + alpha_t * gsl_vector_get(mu,j));
        gsl_matrix_get_col (acol, mata, j);
        gsl_vector_scale(acol, alpha_t*gsl_vector_get(mu,j));
        gsl_vector_sub(mu, acol);
    }
    gsl_vector_free(acol);
    
    gsl_vector_free(delta_b);
    gsl_vector_free(delta_aw);
    gsl_vector_free(d);
    gsl_vector_free(nzs);
    
    return 0;
}

/*****************below are some util functions used in above algorithms*******************/

int updated_svd_algorithms_sp(gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const double xi, int r, int change_r, int * updated_svd) {
    //update d vector
    gsl_vector *d = gsl_vector_calloc(x_t->size);
    gsl_blas_spdaxpy(-gamma_tp1, x_tp1, x_tp1_1s, d);
    gsl_spvector_add(d, x_t, x_t_1s);

    //normalize A matrix and d vector
    double beta = (double)change_r/(1.0+change_r);
    gsl_vector_scale(*vecs, beta);

    //a more compact way to handle the first round?
    int indicator = 0;
    if (change_r == 0) {
        gsl_vector *p  = gsl_vector_alloc(z->size);
        gsl_vector_memcpy(p, z);
        gsl_vector *q = gsl_vector_alloc(d->size);
        gsl_vector_memcpy(q, d);
        double normp = gsl_blas_dnrm2 (p);
        double normq = gsl_blas_dnrm2 (q);
        //set Sigma
        double sigma = normp * normq;
        gsl_vector_set(*vecs, 0, sigma);
        //printf("the first step sigma in tLSTD is:%f\n", sigma);
        //set U and V
        if (normp > 0.00001) {
            gsl_vector_scale(p, 1.0/normp);
        }
        if (normq > 0.00001) {
            gsl_vector_scale(q, 1.0/normq);
        }
        gsl_matrix_set_col(*matu, 0, p);
        gsl_matrix_set_col(*matv, 0, q);
        
        gsl_vector_free(p);
        gsl_vector_free(q);
    }
    else
        indicator = Update_svd(matu, vecs, matv, matl, matr, z, d, r, change_r);

    *updated_svd = 1;

    gsl_vector_free(d);

    return indicator;
}


void update_bvec(gsl_vector * b,gsl_vector * z,const double reward,int change_r) {
    //updated  b vector
  double beta = change_r/(1.0 + change_r);    
    gsl_vector_scale(b, beta);
    gsl_blas_daxpy(reward*(1.0-beta),z, b);
}


int Update_batch_svd(gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix * matz, gsl_matrix * matd, int r, int k, int count){
   
    if(count >= 0){ 
    double beta = 1.0/(double)(count + k);
    gsl_matrix_scale(matd, sqrt(beta));
    gsl_matrix_scale(matz, sqrt(beta));
    }
    //matz size2 be k, size of batch, maybe different with rank r
    //compute Q_z and R_z
    gsl_matrix *work_utz = gsl_matrix_alloc((*matu)->size2, k);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, *matu, matz, 0.0, work_utz);
    gsl_matrix *work_uz = gsl_matrix_alloc((*matu)->size1, k);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *matu, work_utz, 0.0, work_uz);
    gsl_matrix_sub(work_uz, matz);
    gsl_matrix_scale(work_uz, -1);
    
    gsl_matrix *Qz = gsl_matrix_calloc(work_uz->size1, k);
    gsl_matrix *Rz = gsl_matrix_calloc(k, k);
    gsl_vector *tau = gsl_vector_calloc(matz->size2);
    gsl_linalg_QR_decomp (work_uz, tau);
    gsl_matrix_set_identity(Qz);
    gsl_linalg_QR_Qmat (work_uz, tau, Qz);
    gsl_linalg_QR_unpackR (work_uz, tau, k, Rz);
    
    //compute Qd and Rd
    gsl_matrix *work_vtd = gsl_matrix_alloc((*matv)->size2, k);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, *matv, matd, 0.0, work_vtd);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *matv, work_vtd, 0.0, work_uz);
    gsl_matrix_sub(work_uz, matd);
    gsl_matrix_scale(work_uz, -1);
    
    gsl_matrix *Qd = gsl_matrix_calloc(work_uz->size1, k);
    gsl_matrix *Rd = gsl_matrix_calloc(k, k);
    gsl_vector_set_zero(tau);
    gsl_linalg_QR_decomp (work_uz, tau);
    gsl_matrix_set_identity(Qd);
    gsl_linalg_QR_Qmat (work_uz, tau, Qd);
    gsl_linalg_QR_unpackR (work_uz, tau, k, Rd);
    
    gsl_matrix_free(work_uz);
    gsl_vector_free(tau);
    
    //compute K matrix
    int kn = (int)(*matu)->size2 + k;
    int uncol = (*matu)->size2;
    gsl_matrix *matk = gsl_matrix_calloc(kn, kn);
    //divide to four blocks to compute k
    gsl_matrix_view kul = gsl_matrix_submatrix(matk, 0, 0, uncol, uncol);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, work_utz, work_vtd, 0.0, &kul.matrix);
    gsl_matrix_view kll = gsl_matrix_submatrix(matk, uncol, 0, k, uncol);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Rz, work_vtd, 0.0, &kll.matrix);
    gsl_matrix_view kur = gsl_matrix_submatrix(matk, 0, uncol, uncol, k);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, work_utz, Rd, 0.0, &kur.matrix);
    gsl_matrix_view klr = gsl_matrix_submatrix(matk, uncol, uncol, k, k);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Rz, Rd, 0.0, &klr.matrix);
    
    gsl_vector_view diagk = gsl_matrix_diagonal (matk);
    gsl_vector_view diagsubk = gsl_vector_subvector(&diagk.vector, 0, (*vecs)->size);
    gsl_vector_add (&diagsubk.vector, *vecs);
    
    gsl_matrix_free(work_utz);
    gsl_matrix_free(work_vtd);
    gsl_matrix_free(Rz);
    gsl_matrix_free(Rd);
    
    //do SVD on K
    gsl_matrix * Rhat = gsl_matrix_calloc(kn,kn);
    gsl_vector * S = gsl_vector_calloc(kn);
    gsl_vector * work = gsl_vector_calloc(kn);
    
    //A = U S V^T, K = Lhat S Rhat^T, matk becomes Lhat
    gsl_linalg_SV_decomp (matk,  Rhat, S, work);
    gsl_vector_free(*vecs);
    *vecs = S;
    gsl_matrix * Lhat = matk;
    gsl_vector_free(work);
    
    //update Umat and Vmat
    //lu means Left matrix upper part
    gsl_matrix * newmatu = gsl_matrix_alloc((*matu)->size1, Lhat->size2);
    gsl_matrix * newmatv = gsl_matrix_alloc((*matv)->size1, Lhat->size2);
    
    gsl_matrix_view lu = gsl_matrix_submatrix(Lhat, 0, 0, uncol, Lhat->size2);
    gsl_matrix_view ll = gsl_matrix_submatrix(Lhat, uncol, 0, k, Lhat->size2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *matu, &lu.matrix, 0.0, newmatu);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Qz, &ll.matrix, 1.0, newmatu);

    //ru means R matrix upper part
    gsl_matrix_view ru = gsl_matrix_submatrix(Rhat, 0, 0, uncol, Rhat->size2);
    gsl_matrix_view rl = gsl_matrix_submatrix(Rhat, uncol, 0, k, Rhat->size2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *matv, &ru.matrix, 0.0, newmatv);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Qd, &rl.matrix, 1.0, newmatv);

    gsl_matrix_free(*matu);
    gsl_matrix_free(*matv);
    //update u and v matrix
    *matu = newmatu;
    *matv = newmatv;
    
    if ((*vecs)->size >= 2*r) {
        gsl_vector_view subvecs_view = gsl_vector_subvector (*vecs, 0, r);
        gsl_vector *tempvecs = gsl_vector_alloc(r);
        gsl_vector_memcpy (tempvecs, &subvecs_view.vector);
        gsl_vector_free(*vecs);
        *vecs = tempvecs;
        
        //update u matrix
        gsl_matrix_view submatu_view = gsl_matrix_submatrix (*matu, 0, 0, (*matu)->size1, r);
        gsl_matrix *tempmatu = gsl_matrix_alloc((*matu)->size1, r);
        gsl_matrix_memcpy (tempmatu, &(submatu_view.matrix));
        gsl_matrix_free(*matu);
        *matu = tempmatu;
        
        //update v matrix
        gsl_matrix_view submatv_view = gsl_matrix_submatrix (*matv, 0, 0, (*matv)->size1, r);
        gsl_matrix *tempmatv = gsl_matrix_alloc((*matv)->size1, r);
        gsl_matrix_memcpy (tempmatv, &(submatv_view.matrix));
        gsl_matrix_free(*matv);
        *matv = tempmatv;
    }
    
    gsl_matrix_free(matk);
    gsl_matrix_free(Rhat);
    gsl_matrix_free(Qd);
    gsl_matrix_free(Qz);
    
    gsl_matrix_set_zero(matd);
    gsl_matrix_set_zero(matz);
    
    return 0;
}

//if count less than zero, skip normalization
int Update_svd(gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const gsl_vector * z, const gsl_vector * d, int r, int count){
  //compute vector m, first create vector m
  gsl_vector * m = gsl_vector_alloc ((*matl)->size2);
  gsl_vector * tempm = gsl_vector_alloc((*matu)->size2);
  gsl_blas_dgemv (CblasTrans, 1.0, *matu, z, 0.0, tempm);
  gsl_blas_dgemv (CblasTrans, 1.0, *matl, tempm, 0.0, m);
  gsl_vector_free(tempm);
    
  //compute p
  gsl_vector * p = gsl_vector_alloc(z->size);
  gsl_vector_memcpy (p, z);
  gsl_vector * lm = gsl_vector_alloc((*matl)->size1);
  gsl_vector * ulm = gsl_vector_alloc (z->size);
  gsl_blas_dgemv (CblasNoTrans, 1.0, *matl, m, 0.0, lm);
  gsl_blas_dgemv (CblasNoTrans, 1.0, *matu, lm, 0.0, ulm);
  gsl_vector_sub(p,ulm);
  gsl_vector_free(lm);
  gsl_vector_free(ulm);
    
  //compute vector n, first create vector n, similar process with computing m
  gsl_vector * n = gsl_vector_alloc ((*matr)->size2);
  gsl_vector * tempn = gsl_vector_alloc((*matv)->size2);
  gsl_blas_dgemv (CblasTrans, 1.0, *matv, d, 0.0, tempn);
  gsl_blas_dgemv (CblasTrans, 1.0, *matr, tempn, 0.0, n);
  gsl_vector_free(tempn);
    
  //compute q
  gsl_vector * q = gsl_vector_alloc(d->size);
  gsl_vector_memcpy(q, d);
  gsl_vector * rn = gsl_vector_alloc ((*matr)->size1);
  gsl_vector * vrn = gsl_vector_alloc (d->size);
  gsl_blas_dgemv (CblasNoTrans, 1.0, *matr, n, 0.0, rn);
  gsl_blas_dgemv (CblasNoTrans, 1.0, *matv, rn, 0.0, vrn);
  gsl_vector_sub(q, vrn);
  gsl_vector_free(rn);
  gsl_vector_free(vrn);
    
  double normp = gsl_blas_dnrm2 (p);
  double normq = gsl_blas_dnrm2 (q);
    
  //compute K matrix
  gsl_vector * m_normp = gsl_vector_calloc (m->size + 1);
  gsl_vector * n_normq = gsl_vector_calloc (n->size + 1);
  gsl_vector_view subm_norm = gsl_vector_subvector(m_normp, 0, m_normp->size-1);
  gsl_vector_view subn_norm = gsl_vector_subvector(n_normq, 0, n_normq->size-1);
  gsl_vector_memcpy(&subm_norm.vector, m);
  gsl_vector_memcpy(&subn_norm.vector, n);
  gsl_vector_set (m_normp, m_normp->size - 1, normp);
  gsl_vector_set (n_normq, n_normq->size - 1, normq);
  if(count >=0 ){
  double beta = (double)count/(1.0 + (double)count);
  gsl_vector_scale(n_normq, sqrt(1.0-beta));
  gsl_vector_scale(m_normp, sqrt(1.0-beta));
  }
  //compute K
  //printf("reach computing k\n");
  int i, j;
  gsl_matrix * matk = gsl_matrix_calloc(m_normp->size, n_normq->size);
  gsl_vector * temprow = gsl_vector_alloc(matk->size2);
  for(i = 0; i < matk->size1; i++){
      gsl_vector_memcpy(temprow, n_normq);
      gsl_vector_scale(temprow, gsl_vector_get(m_normp,i));
      gsl_matrix_set_row(matk, i, temprow);
  }
  gsl_vector_free(temprow);
  gsl_vector_view diag = gsl_matrix_diagonal (matk);
  //gsl_vector_add_constant(&diag.vector, 0.00001);
  gsl_vector_view diagsub = gsl_vector_subvector(&diag.vector, 0, (*vecs)->size);
  gsl_vector_add (&diagsub.vector, *vecs);
  
  int k = matk->size1;
  gsl_matrix * Rhat = gsl_matrix_calloc(k,k);
  gsl_vector * S = gsl_vector_calloc(k);
  gsl_vector * work = gsl_vector_calloc(k);

  //A = U S V^T, K = Lhat S Rhat^T, matk becomes Lhat
  int status;
  gsl_set_error_handler_off();
  status = gsl_linalg_SV_decomp (matk,  Rhat, S, work);
    if (status) {
        printf("svd error occurred\n");
        diag = gsl_matrix_diagonal (matk);
        gsl_vector_add_constant(&diag.vector, 0.00001);
        status = gsl_linalg_SV_decomp (matk,  Rhat, S, work);
        if (status) {
            printf("svd error occurred twice!!!!\n");
            return 3;
        }
    }
  gsl_vector_free(*vecs);
  *vecs = S;
  gsl_matrix * Lhat = matk;
  
  //update matl and matr
  gsl_matrix * newmatl = gsl_matrix_calloc(k,k);
  gsl_matrix * newmatr = gsl_matrix_calloc(k,k);
  gsl_matrix_set_identity(newmatl);
  gsl_matrix_set_identity(newmatr);
  gsl_matrix_view matl_in_new = gsl_matrix_submatrix(newmatl, 0, 0, (*matl)->size1, (*matl)->size2);
  gsl_matrix_view matr_in_new = gsl_matrix_submatrix(newmatr, 0, 0, (*matr)->size1, (*matr)->size2);
  gsl_matrix_memcpy (&matl_in_new.matrix, *matl);
  gsl_matrix_memcpy (&matr_in_new.matrix, *matr);
  
  gsl_matrix * temp_newmatl = gsl_matrix_alloc(newmatl->size1, newmatl->size2);
  gsl_matrix * temp_newmatr = gsl_matrix_alloc(newmatr->size1, newmatr->size2);
    
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, newmatl, Lhat, 0.0, temp_newmatl);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, newmatr, Rhat, 0.0, temp_newmatr);
  gsl_matrix_free (*matl);
  gsl_matrix_free (*matr);
  gsl_matrix_free(newmatl);
  gsl_matrix_free(newmatr);
  *matl = temp_newmatl;
  *matr = temp_newmatr;
    if (normp > 0.00001) {
        gsl_vector_scale(p, 1.0/normp);
        //printf("p is not too small !!!!!!!!!!!\n");
    }
    else gsl_vector_set_zero(p);
    if (normq > 0.00001) {
        gsl_vector_scale(q, 1.0/normq);
        //printf("q is not too small !!!!!!!!!!!\n");
    }
    else gsl_vector_set_zero(q);
      gsl_matrix * newmatu = gsl_matrix_calloc((*matu)->size1, ((*matu)->size2) + 1);
      gsl_matrix * newmatv = gsl_matrix_calloc((*matv)->size1, ((*matv)->size2) + 1);
      gsl_matrix_view matu_in_new = gsl_matrix_submatrix(newmatu, 0, 0, (*matu)->size1, (*matu)->size2);
      gsl_matrix_view matv_in_new = gsl_matrix_submatrix(newmatv, 0, 0, (*matv)->size1, (*matv)->size2);
      gsl_matrix_memcpy (&matu_in_new.matrix, *matu);
      gsl_matrix_memcpy (&matv_in_new.matrix, *matv);
      j = newmatu->size2 - 1;
      gsl_matrix_set_col(newmatu, j, p);
      gsl_matrix_set_col(newmatv, j, q);
      gsl_matrix_free(*matu);
      gsl_matrix_free(*matv);
      *matu = newmatu;
      *matv = newmatv;
    //}
    int indicator = 0;
  if((*matl)->size2 >= 2*r){
      //update Sigma matrix
      //r = 1;
      gsl_vector_view subvecs_view = gsl_vector_subvector (*vecs, 0, r);
      gsl_vector *tempvecs = gsl_vector_alloc(r);
      gsl_vector_memcpy (tempvecs, &(subvecs_view.vector));
      gsl_vector_free(*vecs);
      *vecs = tempvecs;
      
      //update u matrix
      gsl_matrix * tempu = gsl_matrix_alloc((*matu)->size1, (*matu)->size2);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *matu, *matl, 0.0, tempu);
      gsl_matrix_view submatu_view = gsl_matrix_submatrix (tempu, 0, 0, tempu->size1, r);
      gsl_matrix *tempmatu = gsl_matrix_alloc((tempu)->size1, r);
      gsl_matrix_memcpy (tempmatu, &(submatu_view.matrix));
      gsl_matrix_free(*matu);
      gsl_matrix_free(tempu);
      *matu = tempmatu;
      
	  //update v matrix
      gsl_matrix * tempv = gsl_matrix_alloc((*matv)->size1, (*matv)->size2);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, *matv, *matr, 0.0, tempv);
      gsl_matrix_view submatv_view = gsl_matrix_submatrix (tempv, 0, 0, tempv->size1, r);
      gsl_matrix *tempmatv = gsl_matrix_alloc(tempv->size1, r);
      gsl_matrix_memcpy (tempmatv, &(submatv_view.matrix));
      gsl_matrix_free(*matv);
      gsl_matrix_free(tempv);
      *matv = tempmatv;
	  
	  //reinitialize L and gsl_matrix_free (*matl);
      gsl_matrix_free (*matl);
      gsl_matrix_free (*matr);
      gsl_matrix *tempmatl = gsl_matrix_calloc(r,r);
      gsl_matrix *tempmatr = gsl_matrix_calloc(r,r);
      gsl_matrix_set_identity (tempmatl);
      gsl_matrix_set_identity (tempmatr);
      *matl = tempmatl;
      *matr = tempmatr;
      
      indicator = 1;
  }
    
  gsl_vector_free(m);
  gsl_vector_free(n);
  gsl_vector_free(p);
  gsl_vector_free(q);
  gsl_vector_free(work);
  gsl_vector_free(m_normp);
  gsl_vector_free(n_normq);
  gsl_matrix_free(matk);
  gsl_matrix_free(Rhat);
  return indicator;
}

//update trace
int accumulate_trace(gsl_vector * z, const double lambda_t, const double gamma_t, const gsl_vector * x_t, const gsl_vector * x_t_1s){
    gsl_vector_scale(z, gamma_t*lambda_t);
    gsl_spvector_add(z, x_t, x_t_1s);
    return 0;
}

gsl_vector * compute_dvec(const double gamma_tp1, const gsl_vector * x_t, const gsl_vector * x_t_1s, const gsl_vector* x_tp1, const gsl_vector* x_tp1_1s){
    gsl_vector * d = gsl_vector_calloc(x_t->size);
    gsl_blas_spdaxpy(-gamma_tp1, x_tp1, x_tp1_1s, d);
    gsl_spvector_add(d, x_t, x_t_1s);
    return d;
}


int update_svd_1ststep(const gsl_vector *z, const gsl_vector *d, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv){
    gsl_vector *p  = gsl_vector_alloc(z->size);
    gsl_vector_memcpy(p, z);
    gsl_vector *q = gsl_vector_alloc(d->size);
    gsl_vector_memcpy(q, d);
    double normp = gsl_blas_dnrm2 (p);
    double normq = gsl_blas_dnrm2 (q);
    //set Sigma
    double sigma = normp * normq;
    gsl_vector_set(*vecs, 0, sigma);
    //set U and V
    if (normp > 0.00001) {
        gsl_vector_scale(p, 1.0/normp);
    }
    else gsl_vector_set_zero(p);
    if (normq > 0.00001) {
        gsl_vector_scale(q, 1.0/normq);
    }
    else gsl_vector_set_zero(q);
    gsl_matrix_set_col(*matu, 0, p);
    gsl_matrix_set_col(*matv, 0, q);
    
    gsl_vector_free(p);
    gsl_vector_free(q);
    return 0;
}
