#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "algorithm_utils.h"



/***** Private functions used below, in alphabetical order *****/

void compute_diagonal_inverse(gsl_vector * inv_vec, const gsl_vector * vec, const double threshold);
void multiply_submatrix(gsl_matrix * mat, gsl_matrix * mult, int current_r, gsl_matrix * work);
double normalize_singular_vector(gsl_vector * p);
void point_to_r_submatrices(struct matrix_vars_t * mvars);
int update_K_L_R(struct matrix_vars_t * mvars, double beta, double normp, double normq);
int update_svd(struct matrix_vars_t * mvars, const gsl_vector * z, const gsl_vector * dvec, const double beta);

/***** End Private functions used below ************************/

struct matrix_vars_t * allocate_matrix_vars(const char *mat_type, int numfeatures,  const struct mvar_params_t * mvar_params) {
      struct matrix_vars_t * mvars = malloc(sizeof(struct matrix_vars_t));
      strcpy(mvars->mat_type, mat_type);
      mvars->t = 0;
     //printf("entered alloc %s\n", mat_type); 
     if (strcmp(mat_type, "low_rank") == 0) {
         mvars->r = mvar_params->r;
         mvars->max_r = mvar_params->max_r;
         mvars->threshold = mvar_params->threshold;

         mvars->bvec = gsl_vector_calloc(numfeatures);
         mvars->sigmavec = gsl_vector_calloc(mvars->max_r);
         mvars->matu = gsl_matrix_calloc(numfeatures, mvars->max_r);
         mvars->matv = gsl_matrix_calloc(numfeatures, mvars->max_r);
         mvars->matl = gsl_matrix_alloc(mvars->max_r, mvars->max_r);
         mvars->matr = gsl_matrix_alloc(mvars->max_r, mvars->max_r);
         gsl_matrix_set_identity(mvars->matl);
         gsl_matrix_set_identity(mvars->matr);
         // These work variables are initialized to the maximum size of 2r
         // When using them, we take a subset of the values up to 2r
         mvars->work_d = gsl_vector_calloc(numfeatures);
         mvars->dvec = gsl_vector_calloc(numfeatures);
         mvars->mvec = gsl_vector_calloc(mvars->max_r);
         mvars->nvec = gsl_vector_calloc(mvars->max_r);
         mvars->work_r = gsl_vector_calloc(mvars->max_r);
         mvars->work_r_2 = gsl_vector_calloc(mvars->max_r);
         mvars->work_mat = gsl_matrix_calloc(mvars->max_r,mvars->max_r);
         mvars->Kmat = gsl_matrix_calloc(mvars->max_r,mvars->max_r);
         mvars->Rhat = gsl_matrix_calloc(mvars->max_r,mvars->max_r);
         mvars->current_r = 1;
         point_to_r_submatrices(mvars);
      }
      else if(strcmp(mat_type, "full") == 0){
         mvars->mat_main = gsl_matrix_calloc(numfeatures, numfeatures);
         mvars->delta_main = gsl_matrix_calloc(numfeatures, numfeatures);
         mvars->work_mat_main = gsl_matrix_calloc(numfeatures, numfeatures);
         mvars->work = gsl_vector_calloc(numfeatures);
         mvars->work1 = gsl_vector_calloc(numfeatures);
         mvars->bvec = gsl_vector_calloc(numfeatures);
         mvars->dvec = gsl_vector_calloc(numfeatures);
      }
      else if(strcmp(mat_type, "fullinv") == 0){
         mvars->mat_main = gsl_matrix_calloc(numfeatures, numfeatures);
         mvars->work = gsl_vector_calloc(numfeatures);
         mvars->work1 = gsl_vector_calloc(numfeatures);
         mvars->bvec = gsl_vector_calloc(numfeatures);
         mvars->dvec = gsl_vector_calloc(numfeatures);
      }
     return mvars;
}

void deallocate_matrix_vars(struct matrix_vars_t * mvars) {
      if (strcmp(mvars->mat_type, "full") == 0) {
         gsl_matrix_free(mvars->mat_main);
         gsl_matrix_free(mvars->delta_main);
         gsl_matrix_free(mvars->work_mat_main);
         gsl_vector_free(mvars->work);
         gsl_vector_free(mvars->work1);
         gsl_vector_free(mvars->bvec);
         gsl_vector_free(mvars->dvec);
      }
      else if(strcmp(mvars->mat_type, "fullinv") == 0) {
         gsl_matrix_free(mvars->mat_main);
         gsl_vector_free(mvars->work);
         gsl_vector_free(mvars->work1);
         gsl_vector_free(mvars->bvec);
         gsl_vector_free(mvars->dvec);
      }
      else if(strcmp(mvars->mat_type, "low_rank") == 0){
         gsl_vector_free(mvars->bvec);
         gsl_vector_free(mvars->sigmavec);
         gsl_matrix_free(mvars->matu);
         gsl_matrix_free(mvars->matv);
         gsl_matrix_free(mvars->matl);
         gsl_matrix_free(mvars->matr);

         gsl_vector_free(mvars->work_d);
         gsl_vector_free(mvars->dvec);
         gsl_vector_free(mvars->mvec);
         gsl_vector_free(mvars->nvec);
         gsl_vector_free(mvars->work_r);
         gsl_vector_free(mvars->work_r_2);
         gsl_matrix_free(mvars->work_mat);
         gsl_matrix_free(mvars->Kmat);
         gsl_matrix_free(mvars->Rhat);
      }
    free(mvars);
}

void reset_matrix_vars(struct matrix_vars_t * mvars) {
      if (strcmp(mvars->mat_type, "full") ==0) {
         gsl_matrix_set_zero(mvars->mat_main);
         gsl_matrix_set_zero(mvars->delta_main);
         gsl_matrix_set_zero(mvars->work_mat_main);
         gsl_vector_set_zero(mvars->work);
         gsl_vector_set_zero(mvars->work1);
         gsl_vector_set_zero(mvars->bvec);
         gsl_vector_set_zero(mvars->dvec);
      }
      else if(strcmp(mvars->mat_type, "fullinv") ==0){
         gsl_matrix_set_zero(mvars->mat_main);
         gsl_vector_set_zero(mvars->work);
         gsl_vector_set_zero(mvars->work1);
         gsl_vector_set_zero(mvars->bvec);
         gsl_vector_set_zero(mvars->dvec);
      }
      // Reset the full matrices up to max_r
      else if (strcmp(mvars->mat_type, "low_rank") ==0){
         mvars->current_r = mvars->max_r;
         point_to_r_submatrices(mvars);
         gsl_vector_set_zero(mvars->bvec);
         gsl_vector_set_zero(mvars->sigmavec);
         gsl_matrix_set_zero(mvars->matu);
         gsl_matrix_set_zero(mvars->matv);
         gsl_matrix_set_identity(mvars->matl);
         gsl_matrix_set_identity(mvars->matr);
         
         gsl_vector_set_zero(mvars->work_d);
         gsl_vector_set_zero(mvars->dvec);
         gsl_vector_set_zero(mvars->mvec);
         gsl_vector_set_zero(mvars->nvec);
         gsl_vector_set_zero(mvars->work_r);
         gsl_vector_set_zero(mvars->work_r_2);
         gsl_matrix_set_zero(mvars->work_mat);
         gsl_matrix_set_zero(mvars->Kmat);
         gsl_matrix_set_zero(mvars->Rhat);
         // Now start counting from zero again
         mvars->current_r = 1;
         point_to_r_submatrices(mvars);
      }
      // Now start counting from zero again
      mvars->t = 0;      
}

char * params_to_string(char param_string[MAX_PARAM_STRING_LENGTH], const struct alg_params_t * params) {
      memset(param_string, 0, MAX_PARAM_STRING_LENGTH);
      sprintf(param_string, "%f,%f,%f", params->alpha_t, params->lambda_t, params->eta_t);

      return param_string;
}

void set_params(struct alg_params_t * dest_params, const struct alg_params_t * src_params) {
      dest_params->alpha_t = src_params->alpha_t;
      dest_params->lambda_t = src_params->lambda_t;
      dest_params->beta_t = src_params->beta_t;
      dest_params->lambda_tp1 = src_params->lambda_tp1;
      dest_params->eta_t = src_params->eta_t;
      dest_params->threshold = src_params->threshold;
}

void update_trace_accumulating(gsl_vector * e, const struct alg_params_t * params, const struct transition_info_t * info) {
      gsl_vector_scale(e,info->gamma_t*params->lambda_t);
      gsl_vector_add(e,info->x_t);
}

/* Only makes sense for sparse features; assumes sparse x_t */
void update_trace_replacing(gsl_vector * e, const struct alg_params_t * params, const struct transition_info_t * info) {
      gsl_vector_scale(e,info->gamma_t*params->lambda_t);
      int i;
      for (i = 0; i < info->x_t->size; i++) {
         double new_ei = gsl_vector_get(e, i);
         new_ei = gsl_vector_get(info->x_t, i)> new_ei? gsl_vector_get(info->x_t, i) : new_ei;
         gsl_vector_set(e, i, new_ei);
      }
}

void update_trace_trueonline(gsl_vector * e, const struct alg_params_t * params, const struct transition_info_t * info) {
      double dot;
      gsl_blas_ddot(e, info->x_t, &dot);
  
      double a = params->alpha_t*(1.0 - info->gamma_t*params->lambda_t*dot);
      gsl_vector_scale(e,info->gamma_t*params->lambda_t);
      gsl_blas_daxpy (a, info->x_t, e);  
}

// THIS function is currently NOT used with function pointer
void update_trace_to_gtd(gsl_vector * e, const struct alg_params_t * params, const struct transition_info_t * info) {
      double dot;
      gsl_blas_ddot(e, info->x_t, &dot);
    
      double a = params->alpha_t*(1.0 - info->rho_t*info->gamma_t*params->lambda_t*dot);
      gsl_vector_scale(e,info->gamma_t*params->lambda_t);
      gsl_blas_daxpy (a, info->x_t, e);
      gsl_vector_scale(e, info->rho_t);
}

// DOES NOT UPDATE time step in mvars
// TODO: find a better way to make it clear that update_Amat keeps all the other
// variables in mvars up to date
// potentially by passing only bvec and t to this function, so that its
// clear that mvars could not be updated
void update_bvec(gsl_vector * z, const struct transition_info_t * info, struct matrix_vars_t * mvars){
      double beta = mvars->t/(1.0 + mvars->t);
      gsl_vector_scale(mvars->bvec, beta);
      gsl_blas_daxpy(info->reward*(1.0 - beta), z, mvars->bvec);
}

void update_mat_svd(gsl_vector * z, gsl_vector *d, const struct transition_info_t * info, struct matrix_vars_t * mvars){
      // On first time step, no svd, closed form update
      if (mvars->t == 0) {
         // p = z
         gsl_vector_memcpy(mvars->work_d, z);
         double normp = normalize_singular_vector (mvars->work_d);
         gsl_matrix_set_col(mvars->matu, 0, mvars->work_d);
         // q = d
         gsl_vector_memcpy(mvars->work_d, d);
         //printLine;
         double normq = normalize_singular_vector (mvars->work_d);
         gsl_matrix_set_col(mvars->matv, 0, mvars->work_d);         

         gsl_vector_set(mvars->sigmavec, 0, normp * normq);

         mvars->current_r = 1;
      }
      else {
         // keep normalized A matrix using beta = t/(1+t)
         double beta = (double)mvars->t/(1.0 + (double)mvars->t);
         //printf("the current beta is %f\n", beta);
         gsl_vector_scale(mvars->sigmavec, beta);

         /*
          * Have a separate work variable for dvec, to ensure that it is not overwritten
          * in update_svd. Cannot use mvars->work_d for dvec, as that will be overwritten
          */
         update_svd(mvars, z, d, beta);
         
      }

      // Ensure all the pointers matu, matv, etc. are pointing to the correct current_r size matrices
      point_to_r_submatrices(mvars);
     
      mvars->t++;
}

//NOTE: sherman update use normalization
void update_mat_sherman(gsl_vector * z, gsl_vector *d, const struct transition_info_t * info, struct matrix_vars_t * mvars){
    
      //update A matrix
      gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->mat_main, z, 0, mvars->work);
      double denominator = 0;
      gsl_blas_ddot(d, mvars->work, &denominator);
      
      //denominator+=1;
      if(mvars->t > 0)denominator = mvars->t*mvars->t + mvars->t * denominator;
      else denominator += 1;

      gsl_blas_dgemv (CblasTrans, 1.0, mvars->mat_main, d, 0, mvars->work1);
    
      //gsl_blas_dger (-1.0/denominator, mvars->work, mvars->work1, mvars->mat_main);
      
      if(mvars->t > 0)gsl_matrix_scale(mvars->mat_main, (mvars->t + 1.0)/mvars->t);

      gsl_blas_dger (-(mvars->t + 1.0)/denominator, mvars->work, mvars->work1, mvars->mat_main);

      mvars->t++;
}

void update_mat_normal(gsl_vector * z, gsl_vector *d, const struct transition_info_t * info, struct matrix_vars_t * mvars){
    
      double beta = 1.0/(1.0 + (double)mvars->t);
    
      gsl_matrix_scale (mvars->mat_main, 1.0 - beta);
    
      gsl_blas_dger (beta, z, d, mvars->mat_main);
    
      mvars->t++;
}

// w = beta * w + alpha * (matrix_type_op(matrix) * v), matrix is the main matrix in mvars
void update_weights(gsl_vector *w, gsl_vector *v, double alpha, double beta, struct matrix_vars_t * mvars, MAT_TYPE_OP matrix_type_op){
      if (matrix_type_op == MAT_SVD_INV) {
         // 1. V R Sig L^T U^T
         gsl_blas_dgemv (CblasTrans, 1.0, mvars->matu, v, 0.0, mvars->work_r);
         // 2. work_r = L^T work_r  is an r-dim variable
         gsl_blas_dgemv (CblasTrans, 1.0, mvars->matl, mvars->work_r, 0.0, mvars->work_r_2);
         // 3. work_r = Sigmainv * work_r
         compute_diagonal_inverse(mvars->work_r, mvars->sigmavec, mvars->threshold);
         gsl_vector_mul(mvars->work_r_2, mvars->work_r);
         // 4. work_r = R * work_r
         gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matr, mvars->work_r_2, 0.0, mvars->work_r);
         // 4. w = beta* w + alpha * V * work_r
         gsl_blas_dgemv (CblasNoTrans, alpha, mvars->matv, mvars->work_r, beta, w);
      }
      else if(matrix_type_op == MAT_SVD){
         // 1. U L Sig R^T V^T
         gsl_blas_dgemv (CblasTrans, 1.0, mvars->matv, v, 0.0, mvars->work_r);
         gsl_blas_dgemv (CblasTrans, 1.0, mvars->matr, mvars->work_r, 0.0, mvars->work_r_2);
         gsl_vector_mul(mvars->work_r_2, mvars->sigmavec);
         gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matl, mvars->work_r_2, 0.0, mvars->work_r);
         gsl_blas_dgemv (CblasNoTrans, alpha, mvars->matu, mvars->work_r, beta, w);
      }
      else if(matrix_type_op == MAT_SVD_TRANS){
         gsl_blas_dgemv (CblasTrans, 1.0, mvars->matu, v, 0.0, mvars->work_r);
         gsl_blas_dgemv (CblasTrans, 1.0, mvars->matl, mvars->work_r, 0.0, mvars->work_r_2);
         gsl_vector_mul(mvars->work_r_2, mvars->sigmavec);
         gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matr, mvars->work_r_2, 0.0, mvars->work_r);
         gsl_blas_dgemv (CblasNoTrans, alpha, mvars->matv, mvars->work_r, beta, w);
      }
      else if(matrix_type_op == MAT_FULL){
         gsl_blas_dgemv (CblasNoTrans, alpha, mvars->mat_main, v, beta, w);
      }
      else if(matrix_type_op == MAT_FULL_INV){
         gsl_matrix_memcpy(mvars->delta_main, mvars->mat_main);
         gsl_vector_view diag = gsl_matrix_diagonal (mvars->delta_main);
         gsl_vector_add_constant (&diag.vector, 0.00001);
         gsl_linalg_SV_decomp (mvars->delta_main, mvars->work_mat_main, mvars->work, mvars->work1);
         gsl_vector_set_zero(mvars->work1);
         gsl_linalg_SV_solve (mvars->delta_main, mvars->work_mat_main, mvars->work, v,  mvars->work1);
         gsl_vector_scale(w, beta);
         gsl_blas_daxpy(alpha, mvars->work1, w);
      }
}

void compute_dvec(gsl_vector *d, const struct transition_info_t * info){
      gsl_vector_memcpy(d, info->x_tp1);
      gsl_vector_scale(d, -info->gamma_tp1);
      gsl_vector_add(d, info->x_t);
}

double compute_delta(gsl_vector * w, const struct transition_info_t * info){
    
      double old_v = 0;
      gsl_blas_ddot (w, info->x_t, &old_v);
      double new_v = 0;
      gsl_blas_ddot (w, info->x_tp1, &new_v);
    
      double delta = info->reward + info->gamma_tp1*new_v - old_v;
    
      return delta;
}

void compute_weights(gsl_vector * w, struct matrix_vars_t * mvars) {

      // w = V R Sigmainv L^T U^T b; to compute this
      // 1. work_r = U^T b  is an r-dim variable
      gsl_blas_dgemv (CblasTrans, 1.0, mvars->matu, mvars->bvec, 0.0, mvars->work_r);
    
      // 2. work_r = L^T work_r  is an r-dim variable
      gsl_blas_dgemv (CblasTrans, 1.0, mvars->matl, mvars->work_r, 0.0, mvars->work_r_2);
    
      // 3. work_r = Sigmainv * work_r
      compute_diagonal_inverse(mvars->work_r, mvars->sigmavec, mvars->threshold);
      gsl_vector_mul(mvars->work_r_2, mvars->work_r);
    
      // 4. work_r = R * work_r 
      gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matr, mvars->work_r_2, 0.0, mvars->work_r);
    
      // 4. w = V * work_r 
      gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matv, mvars->work_r, 0.0, w);

}

// matrix times a vector: op(matrix)*vector
void op_mat_vector_mul(gsl_vector * res, gsl_vector * b, struct matrix_vars_t * mvars, MAT_TYPE_OP matrix_type_op) {
      if (matrix_type_op == MAT_SVD_INV) {
         gsl_blas_dgemv (CblasTrans, 1.0, mvars->matu, b, 0.0, mvars->work_r);
         gsl_blas_dgemv (CblasTrans, 1.0, mvars->matl, mvars->work_r, 0.0, mvars->work_r_2);
         compute_diagonal_inverse(mvars->work_r, mvars->sigmavec, mvars->threshold);
         gsl_vector_mul(mvars->work_r_2, mvars->work_r);
         gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matr, mvars->work_r_2, 0.0, mvars->work_r);
         gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matv, mvars->work_r, 0.0, res);
      }
      else if(matrix_type_op == MAT_FULL_INV) {
         gsl_matrix_memcpy(mvars->delta_main, mvars->mat_main);
         // gsl_vector_view diag = gsl_matrix_diagonal (mvars->delta_main);
         // gsl_vector_add_constant (&diag.vector, 0.00001);
         gsl_vector_set_zero(mvars->work1);
         gsl_vector_set_zero(mvars->work);
         gsl_matrix_set_zero(mvars->work_mat_main);
         gsl_linalg_SV_decomp (mvars->delta_main, mvars->work_mat_main, mvars->work, mvars->work1);
         gsl_vector_set_zero(res);
         gsl_linalg_SV_solve (mvars->delta_main, mvars->work_mat_main, mvars->work, b,  res);
      }
}

/************************ Private functions for SVD *********************/


// update_svd always ensures that the sizes of submatrix variables is correct
int update_svd(struct matrix_vars_t * mvars, const gsl_vector * z, const gsl_vector * dvec, const double beta){
  
      /*
       * First, compute p = z - (U L) (U L)^T z
       * and set U = [U p/pnorm]
       */
      // compute m = (U L)^T z
      // access memory in mvec up to current_r
      gsl_blas_dgemv (CblasTrans, 1.0, mvars->matu, z, 0.0, mvars->work_r);
      gsl_blas_dgemv (CblasTrans, 1.0, mvars->matl, mvars->work_r, 0.0, mvars->mvec);
    
      // mvars->work_r_2 = L * mvec
      gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matl, mvars->mvec, 0.0, mvars->work_r);
      // mvars->work_d = U *  L * mvec
      gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matu, mvars->work_r, 0.0, mvars->work_d);
      // p = z - mvars->work_d = z - U L mvec
      gsl_vector_scale(mvars->work_d, -1.0);
      gsl_vector_add(mvars->work_d, z);
      double normp = normalize_singular_vector(mvars->work_d);
      // Can update U now, because it does not influence Kmat update
      mvars->matu->size2++;      
      gsl_matrix_set_col(mvars->matu, mvars->current_r, mvars->work_d);

      /*
       * Second, compute q =  d - (V R) (V R)^T d
       * and set V = [V q/qnorm]
       */
      //compute n = (V R)^T d
      gsl_blas_dgemv (CblasTrans, 1.0, mvars->matv, dvec, 0.0, mvars->work_r);
      gsl_blas_dgemv (CblasTrans, 1.0, mvars->matr, mvars->work_r, 0.0, mvars->nvec);

      gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matr, mvars->nvec, 0.0, mvars->work_r);
      gsl_blas_dgemv (CblasNoTrans, 1.0, mvars->matv, mvars->work_r, 0.0, mvars->work_d);
      gsl_vector_scale(mvars->work_d, -1.0);
      gsl_vector_add(mvars->work_d, dvec);
      double normq = normalize_singular_vector(mvars->work_d);
      // Can update V now, because it does not influence Kmat update
      mvars->matv->size2++;
      gsl_matrix_set_col(mvars->matv, mvars->current_r, mvars->work_d);
    
      /*
       * Third, update K and left and right subspace L and R
       * This involves increasing the rank
       * Updating mvec and nvec
       * Adding to get Kmat = beta[Sigma, 0; 0 0] + m n^T
       * And then computing the SVD
       */
      update_K_L_R(mvars, beta, normp, normq);

      /* 
       * Finally, check if going to go past max_r; if so, set current_r = r to truncate back down to r 
       * And multiply UL and VR before doing this aggressive truncation
       */
      int indicator = 0;
      if(mvars->current_r >= mvars->max_r){
         // All vectors below truncated by setting current_r to r, after this if statement
         mvars->current_r = mvars->r;

         // This matrix allocation happens only every (max_r-r) steps, so not worth removing 
         gsl_matrix * temp = gsl_matrix_calloc(mvars->matu->size1, mvars->matu->size2);
         
         // U = U * L and then truncated to U(:, 1:r) by setting mvars->current_r = r
         gsl_matrix_product_A_equal_AB(mvars->matu, mvars->matl, temp);
      
         // V = V * R and then truncated to V(:, 1:r)
         gsl_matrix_product_A_equal_AB(mvars->matv, mvars->matr, temp);
         
         gsl_matrix_free(temp);
         
         // Reset L and R to identity matrices
         gsl_matrix_set_identity (mvars->matl);
         gsl_matrix_set_identity (mvars->matr);
          
         indicator = 1;
      }
    
      return indicator;
}


// Ensure all the variables are pointing to the appropriate point in the full matrices
void point_to_r_submatrices(struct matrix_vars_t * mvars){

      mvars->sigmavec->size = mvars->current_r;
      mvars->matu->size2 = mvars->current_r;
      mvars->matv->size2 = mvars->current_r;
      mvars->matl->size1 = mvars->current_r;
      mvars->matl->size2 = mvars->current_r;
      mvars->matr->size1 = mvars->current_r;
      mvars->matr->size2 = mvars->current_r;

      // Reset pointers for work variables also
      // work variables for numfeatures = d are not modified
      mvars->mvec->size = mvars->current_r;
      mvars->nvec->size = mvars->current_r;
      mvars->work_r->size = mvars->current_r;
      mvars->work_r_2->size = mvars->current_r;
      mvars->work_mat->size1 = mvars->current_r; 
      mvars->work_mat->size2 = mvars->current_r; 
      mvars->Kmat->size1 = mvars->current_r; 
      mvars->Kmat->size2 = mvars->current_r; 
      mvars->Rhat->size1 = mvars->current_r; 
      mvars->Rhat->size2 = mvars->current_r; 
}

/*
 * Returns the norm of p
 * Even if norm below a threshold and p set to zero
 * the returned norm is the original norm of p
 */
double normalize_singular_vector(gsl_vector * p) {

      double normp = gsl_blas_dnrm2 (p);

      if (normp > MIN_SVD_VEC_NORM) {
         gsl_vector_scale(p, 1.0/normp);
      }
      else gsl_vector_set_zero(p);

      return normp;
}

void compute_diagonal_inverse(gsl_vector * inv_vec, const gsl_vector * vec, const double threshold) {
      int i;
      double val;
      for (i = 0; i < vec->size; i++){
         val = 0;
         // TODO: what criteria should we use here?
         //if(gsl_vector_get(vec, i) > max(threshold * gsl_vector_get(vec, 0), threshold))
         if(gsl_vector_get(vec, i) > threshold * gsl_vector_get(vec, 0))
               val = 1.0/gsl_vector_get(vec, i);
         gsl_vector_set(inv_vec, i, val);
      }
}

int update_K_L_R(struct matrix_vars_t * mvars, double beta, double normp, double normq) {
      /*
       * Updating K involves updating mvec and nvec
       * Adding to get Kmat = beta[Sigma, 0; 0 0] + m n^T
       * Computing the SVD
       * And finally updating L and R
       */

      /* 
       * Zero: first increase the size of the required variables by 1
       * since we will be increasing the rank by 1
       */
      mvars->current_r++;
      point_to_r_submatrices(mvars);
     

      /* 
       * First compute mvec and nvec
       */
      // Increase dimension of mvec to incorporate normp; nvec with normq
      gsl_vector_set(mvars->mvec, mvars->current_r-1, normp);
      gsl_vector_set(mvars->nvec, mvars->current_r-1, normq);
      // Scale m and n with beta
      double scale = 1.0-beta > 0.001? 1.0-beta: 0.001;
      scale = 1.0 - beta;
      gsl_vector_scale(mvars->mvec, sqrt(scale));
      gsl_vector_scale(mvars->nvec, sqrt(scale));
  
      /* 
       * Second set Kmat = beta[Sigma, 0; 0 0] + m n^T
       */
      gsl_matrix_set_zero(mvars->Kmat);
      // do not need to check subset of mvec and nvec to current_r,
      // since the zero parts of mvec, nvec just add more zeros to Kmat
      gsl_blas_dger (1.0, mvars->mvec, mvars->nvec, mvars->Kmat);
      for(int i = 0; i < mvars->current_r-1; i++){
         gsl_matrix_set(mvars->Kmat, i, i, gsl_matrix_get(mvars->Kmat, i,i) + gsl_vector_get(mvars->sigmavec,i));
      }
  
      /*
       * Third compute the SVD of K
       */

      //A = U S V^T, K = Lhat S Rhat^T, Kmat becomes Lhat, work_r = S
      int status;
      gsl_set_error_handler_off();
      status = gsl_linalg_SV_decomp (mvars->Kmat,  mvars->Rhat, mvars->work_r, mvars->work_r_2);
      if (status) {
         fprintf(stderr, "update_svd -> svd error occurred\n");
         // Recompute Kmat, but add a small positive value to the diagonal
         gsl_matrix_set_zero(mvars->Kmat);
         gsl_blas_dger (1.0, mvars->mvec, mvars->nvec, mvars->Kmat);
         for(int i = 0; i < mvars->current_r-1; i++)
               gsl_matrix_set(mvars->Kmat, i, i, gsl_matrix_get(mvars->Kmat, i,i) + gsl_vector_get(mvars->sigmavec,i));

         gsl_vector_view diag = gsl_matrix_diagonal (mvars->Kmat);
         gsl_vector_add_constant(&diag.vector, 0.00001);
         status = gsl_linalg_SV_decomp (mvars->Kmat,  mvars->Rhat, mvars->work_r, mvars->work_r_2);
         if (status) {
            fprintf(stderr, "update_K_L_R -> svd error occurred twice!!!!\n");
            return 3;
         }
      }
      gsl_vector_memcpy(mvars->sigmavec, mvars->work_r);

      /*
       * Fourth, update L and R 
       * L = [L 0; 0 1] Lhat
       * R = [R 0; 0 1] Rhat
       * where mvars->Kmat now contains Lhat and mvars->Rhat contains Rhat
       */
      gsl_vector_view lastrow = gsl_matrix_row(mvars->matl, mvars->current_r-1);
      gsl_vector_view lastcol = gsl_matrix_column(mvars->matl, mvars->current_r-1);
      gsl_vector_set_zero(&lastrow.vector);
      gsl_vector_set_zero(&lastcol.vector);
      lastrow = gsl_matrix_row(mvars->matr, mvars->current_r-1);
      lastcol = gsl_matrix_column(mvars->matr, mvars->current_r-1);
      gsl_vector_set_zero(&lastrow.vector);
      gsl_vector_set_zero(&lastcol.vector);
      gsl_matrix_set(mvars->matl, mvars->current_r-1, mvars->current_r-1, 1.0);
      gsl_matrix_set(mvars->matr, mvars->current_r-1, mvars->current_r-1, 1.0);
      gsl_matrix_product_A_equal_AB(mvars->matl, mvars->Kmat, mvars->work_mat);
      gsl_matrix_product_A_equal_AB(mvars->matr, mvars->Rhat, mvars->work_mat);

      return 0;
}

// Return A = A * B
int gsl_matrix_product_A_equal_AB(gsl_matrix *A, gsl_matrix *B, gsl_matrix *work){
      gsl_matrix_memcpy(work, A);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, work, B, 0.0, A);
      return 0;
}

