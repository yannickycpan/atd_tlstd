#include <stdlib.h>
#include <stdio.h>

#include "utils.h"

void init_array(double * v, const double val, const int len) {
  for (int i = 0; i < len; i++) {
    v[i] = val;
  }
}

int alloc_vectors(gsl_vector * arr[], const int len1, const int len2){
  for (int i =0; i < len1; i++) {
    arr[i] = gsl_vector_alloc(len2);
  }
  return 0;
}

void zero_vectors(gsl_vector * arr[], const int len1){
  for (int i =0; i < len1; i++) {
    gsl_vector_set_zero(arr[i]);
  }
}

int gsl_vector_contains(gsl_vector * a, const int j){
  for (int i =0; i < a->size; i++) {
    if(gsl_vector_get(a,i) == j)
      return 1;
  }
  return 0;
}

double gsl_vector_sum (gsl_vector * arr){
  double sum=0;
  for (int i =0; i < arr->size; i++) {
    sum = sum + gsl_vector_get(arr,i);
  }
  return sum;
}

int free_vectors(gsl_vector * arr[], const int len1){
  for (int i =0; i < len1; i++) {
     gsl_vector_free(arr[i]);
  }
  return 0;
}

void init_rgen(struct rgen_t * rgen, const unsigned long int seed) {
  const gsl_rng_type * T;
  
  gsl_rng_env_setup ();
  
  T = gsl_rng_default;
  rgen->r = gsl_rng_alloc (T);
  gsl_rng_set(rgen->r, seed);
}

void set_rgen_seed(struct rgen_t * rgen, const unsigned long int seed) {
  gsl_rng_set(rgen->r, seed);
}

char* itoa(int val, int base){
	
	static char buf[32] = {0};
	
	int i = 30;
	
	for(; val && i ; --i, val /= base)
	
		buf[i] = "0123456789abcdef"[val % base];
	
	return &buf[i+1];
	
}


int int2Binary(gsl_vector * array, int inv, int n)
{
  char* buffer;
  gsl_vector_set_all(array,0);
  int nbits = floor(log2(n))+1;
  //printf("convert %d to ",inv);
  if(inv == 0){
    //printf("00000\n");
    return 0;
  }
    
  buffer = itoa (inv,2);
  int len = strlen(buffer);
  int j = nbits-1;
  for(int i =len-1;i>=0;i--){
    if(buffer[i] == '1')
      gsl_vector_set(array,j,1);
      j = j -1;
  }

  return 0;
}

double log2(double a)
{
  return log(a)/log(2);
}

double rand_un()
{
    return (double)rand() / (double)RAND_MAX;
}

double randn (double mu, double sigma)
{
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;
    
    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (double) X2);
    }
    
    do
    {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);
    
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    
    call = !call;
    
    return (mu + sigma * (double) X1);
}

double uniform_random(struct rgen_t * rgen) {
        return gsl_rng_uniform(rgen->r);
        }

double gaussian_random(struct rgen_t * rgen, double sigma, double mean) {
        return gsl_ran_gaussian(rgen->r,sigma) + mean;
        }

int uniform_random_inRange(struct rgen_t * rgen, const int n) {
  return gsl_rng_uniform_int(rgen->r,n);
        }

void free_rgen(struct rgen_t * rgen) {
   gsl_rng_free(rgen->r);
}

int generate_random_vector(gsl_vector * vec, const double sigma, double mean,struct rgen_t * rgen){
    for (int i = 0; i < vec->size; i++) {
        gsl_vector_set(vec, i, gaussian_random(rgen,sigma,mean));
    }
    return 0;
}

int generate_random_uniform_vector(gsl_vector * vec, struct rgen_t * rgen){
    for (int i = 0; i < vec->size; i++) {
        gsl_vector_set(vec, i, uniform_random(rgen));
    }
    return 0;
}

int generate_random_matrix(gsl_matrix * mat, const double sigma, double mean,struct rgen_t * rgen)
{ 
  gsl_matrix_set_zero (mat);
         
  for (int i=0; i<mat->size1; i++) {
    for (int j=0; j<mat->size2; j++){
      gsl_matrix_set (mat, i, j, gaussian_random(rgen,sigma,mean));
    }    
  }
  return 0;
}

int generate_random_uniform_matrix(gsl_matrix * mat, struct rgen_t * rgen, double minV)
{
  
  gsl_matrix_set_zero (mat);

  for (int i=0; i<mat->size1; i++) {
    for (int j=0; j<mat->size2; j++){
      double v =  uniform_random(rgen);
      if(v < minV)
	v = minV;
        gsl_matrix_set (mat, i, j,v);
    }    
  }
  return 0;
}

double min(double a, double b){
  if(a <= b)
    return a;
  return b;
  
}

int generate_random_indices(gsl_vector * v, const int n, struct rgen_t * rgen)
{

  gsl_vector_set_all (v,-1);
  for (int i=0; i<v->size; i++){
    int val = uniform_random_inRange(rgen, n);
    while(gsl_vector_contains(v,val))
      val = uniform_random_inRange(rgen, n);
    gsl_vector_set(v, i, val);
  }
  return 0;
}
int normalize_rows_matrix(gsl_matrix * mat)
{
  for (int i=0; i<mat->size1; i++) {
    double sum=0;
    for (int j=0; j<mat->size2; j++){
      sum = sum + gsl_matrix_get (mat, i, j);
    }
    for (int j=0; j<mat->size2; j++){
      double val = gsl_matrix_get (mat, i, j);
      gsl_matrix_set (mat, i, j, val/sum);
    }        
  }
  return 0;
}

int normalize_rows_matrix_cut(gsl_matrix * mat, const int n)
{
  for (int i=0; i<mat->size1; i++) {
    double sum=0;
    for (int j=0; j<n; j++){
      sum = sum + gsl_matrix_get (mat, i, j);
    }
    for (int j=0; j<n; j++){
      double val = gsl_matrix_get (mat, i, j);
      gsl_matrix_set (mat, i, j, val/sum);
    }        
  }
  return 0;
}


int sample_from_dist_nocumsum(const double * probs, gsl_vector * indexs, struct rgen_t * rgen)
{
  
  double rand_n = uniform_random(rgen);
  int ind=0;
  while(probs[ind]< rand_n)
    ind=ind+1;
    
  return ind;
}

int gsl_vector_print(gsl_vector * v)
{
  for(int i=0;i<v->size;i++)
    printf("%f ",gsl_vector_get(v,i));
  printf("\n");
  return 0;
}

int gsl_matrix_print(gsl_matrix * m)
{
  for(int i=0;i<m->size1;i++){
    gsl_vector_view v = gsl_matrix_row(m,i); 
    gsl_vector_print(&v.vector);
    }
  return 0;
}

int normalize_cols_matrix(gsl_matrix * mat)
{
  for (int i=0; i<mat->size2; i++) {
    double sum=0;
    for (int j=0; j<mat->size1; j++){
      sum = sum + gsl_matrix_get (mat, j, i);
    }
    for (int j=0; j<mat->size1; j++){
      double val = gsl_matrix_get (mat, j, i);
      gsl_matrix_set (mat, j, i, val/sum);
    }        
  }
  return 0;
}

double normdiff(const gsl_vector* v1, const gsl_vector* v2, const int len) {
  double nd = 0, diff = 0;
  for (int i =0; i < len; i++) {
    diff = gsl_vector_get(v1,i)-gsl_vector_get(v2,i);
    nd += diff*diff;
  }
  return nd;
}

double multiply_vec(const double* v1, const double* v2, const int len) {
  double m = 0;
  for (int i =0; i < len; i++) {
    m += v1[i]*v2[i];
  }
  return m;
}

int gsl_matrix_matrix_rect(gsl_matrix *C, gsl_matrix *A, gsl_matrix *B){
  for(int i=0;i<A->size1;i++)
    {
      gsl_vector_view rowA = gsl_matrix_row (A, i);
      for(int j=0;j<B->size2;j++)
      {
	gsl_vector_view colB = gsl_matrix_column (B, j);
	double dot = 0; //multiply_gsl_vec(&rowA.vector, &colB.vector);
	gsl_blas_ddot (&rowA.vector, &colB.vector, &dot); 
	
	/*printf("(%d,%d) dot = %f\n",i,j,dot);*/
	gsl_matrix_set(C,i,j,dot);
      }
    }
  return 0;
}

int gsl_matrix_vector_product(gsl_vector *c, gsl_matrix *A, gsl_vector *b){
  gsl_blas_dgemv (CblasNoTrans, 1.0, A, b, 0, c);

  return 0;
}

double multiply_gsl_vec(gsl_vector *v1, gsl_vector *v2) {
  double m = 0;
  for (int i =0; i < v1->size; i++) {
    m += gsl_vector_get(v1,i)*gsl_vector_get(v2,i);
  }
  return m;
}

int gsl_outer_product(gsl_matrix *A, gsl_vector *v1, gsl_vector *v2){
    if (A->size1 != v1->size || A->size2 != v2->size) {
        printf("Error: Vector-Matrix dimension does not match!");
        return 0;
    }
    gsl_vector *temprow = gsl_vector_alloc(A->size2);
    for (int i = 0; i<v1->size; i++) {
        gsl_vector_memcpy(temprow, v2);
        gsl_vector_scale(temprow, gsl_vector_get(v1, i));
        gsl_matrix_set_row(A, i, temprow);
    }
    gsl_vector_free(temprow);
    return 1;
}

void divide_all( double *vec, const double divisor, const int len) {
  for (int i = 0; i < len; i++) {
    vec[i] = vec[i]/divisor;
  }
}

int write_matrix_to_file(const int n, const int m, double mat[][m], char* name)
{
    FILE *matrix=fopen(name, "w");    

    for(int i=0;i<n;i++)     
    {
      fprintf(matrix, "%d ",(i+1));
        for(int j=0;j<m;j++)  
        {              
            fprintf(matrix, "%f ", mat[i][j]);
        }
        fprintf(matrix, "\n");
    } 
    fclose(matrix);

    return 0;
}

double gsl_get_cond_num(gsl_matrix* A)
{
    gsl_vector * S = gsl_vector_alloc(A->size2);
    gsl_vector * work = gsl_vector_alloc(A->size2);
    gsl_matrix * V = gsl_matrix_alloc(A->size2, A->size2);

    gsl_linalg_SV_decomp (A, V, S, work);
    
    double cond = gsl_vector_max(S)/gsl_vector_min(S);
    
    gsl_vector_free(S);
    gsl_vector_free(work);
    gsl_matrix_free(V);
    return cond;
    
}

double gsl_spectral_radius(gsl_matrix * mat)
{
        gsl_vector_complex *eval = gsl_vector_complex_alloc (mat->size1);
        gsl_matrix_complex *evec = gsl_matrix_complex_alloc (mat->size1, mat->size1);
	
        gsl_eigen_nonsymmv_workspace * w =  gsl_eigen_nonsymmv_alloc (mat->size1);
        gsl_eigen_nonsymmv (mat, eval, evec, w);
        gsl_eigen_nonsymmv_free (w);
        gsl_eigen_nonsymmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_DESC);
	gsl_vector_view vals = gsl_vector_complex_real (eval);

	double eig_max =gsl_vector_get(&vals.vector,0);

	gsl_vector_complex_free (eval);
        gsl_matrix_complex_free (evec);
	
	return eig_max; 
}

int SVD_inv(gsl_matrix * A, int k, double tol)
{
        gsl_matrix * V = gsl_matrix_alloc(k,k);
        gsl_vector * S = gsl_vector_alloc(k);
        gsl_vector * work = gsl_vector_alloc(k);
        gsl_matrix * Smat = gsl_matrix_alloc(k,k);

        //A = U S V^T
        gsl_linalg_SV_decomp (A,  V, S, work);
    
        for(int i=0;i<k;i++)
        {
            double val = 0;
            if(gsl_vector_get(S,i) >= tol*gsl_vector_get(S, 0))
                    val = 1.0/gsl_vector_get(S,i);
            gsl_matrix_set(Smat,i,i,val);
        }

        //Ain = V Sin U^T;
        gsl_matrix * VS = gsl_matrix_alloc(k,k);
    
        gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, V, Smat, 0, VS);
        gsl_matrix_memcpy(V,A);
        gsl_matrix_set_all(A,0);
        gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, VS, V, 0, A);
    
        gsl_matrix_free(V);
        gsl_matrix_free(Smat);
        gsl_matrix_free(VS);
        gsl_vector_free(S);
        gsl_vector_free(work);
    
        return 0;
}

int FiveDto1DIndex(int a, int b, int c, int d, int e, const struct ArraySize* sizes){
  return a + b*sizes->n1 + c*sizes->n1*sizes->n2 + d*sizes->n1*sizes->n2*sizes->n3 + e*sizes->n1*sizes->n2*sizes->n3*sizes->n4;
}

int SixDto1DIndex(int a, int b, int c, int d, int e, int h, const struct ArraySize* sizes){
    return a + b*sizes->n1 + c*sizes->n1*sizes->n2 + d*sizes->n1*sizes->n2*sizes->n3 + e*sizes->n1*sizes->n2*sizes->n3*sizes->n4 + h*sizes->n1*sizes->n2*sizes->n3*sizes->n4*sizes->n5;
}


int write_matrix_to_file_with_names(const int n, const int m, double mat[][m],  const int nNames, char* names[], char* name)
{
    FILE *matrix=fopen(name, "w");
    fprintf(matrix, "#No\t");
    
    for(int i=0;i<nNames;i++)     
      fprintf(matrix, "%s\t",names[i]);
    fprintf(matrix, "\n");
    
    
    for(int i=0;i<n;i++)     
    {
      fprintf(matrix, "%d\t",(i+1));
        for(int j=0;j<m-1;j++)  
        {              
            fprintf(matrix, "%f\t", mat[i][j]);
        }
        fprintf(matrix, "%f\n", mat[i][m-1]);
    } 
    fclose(matrix);

    return 0;
}

int print_matrix(const int n, const int m, double mat[][m])
{
    for(int i=0;i<n;i++)     
    {
      printf("%d ",i);
        for(int j=0;j<m;j++)  
        {              
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    } 
    return 0;
}



int write_vector_to_file(const int n, double* vec, char* name)
{
  FILE *filePtr;
  filePtr = fopen(name,"w");

    for(int i=0;i<n;i++)     
    {            
            fprintf(filePtr, "%f\n", vec[i]);
    }
    fprintf(filePtr, "\n");
    fclose(filePtr);

    return 0;
}

int compute_mean_variance_of_vector(const gsl_vector * vec, double * mean, double * var){
  double sumX=0;
  double sumX2 = 0;
  int len = vec->size;
  for(int i=0;i<len;i++){
    double val = gsl_vector_get(vec,i);
    sumX += val;
    sumX2 += val*val;
  }
  *mean = sumX/len;
  *var = sumX2/len - (sumX/len)*(sumX/len);
  return 0; 
}

void divide_all_mat(const int len1, const int len2, double mat[][len2], const double divisor) {
  for (int i = 0; i < len1; i++) {
    for (int j = 0; j < len2; j++) {
      mat[i][j] = mat[i][j]/divisor;
    }
  }
}

/* Compute error functions */
double compute_rmse(const gsl_vector *p, const gsl_vector *t){
    double err = 0;
    gsl_vector *temp_p = gsl_vector_alloc(p->size);
    gsl_vector_memcpy(temp_p, p);
    gsl_vector_sub(temp_p, t);
    err = gsl_blas_dnrm2 (temp_p);
    err = err/sqrt(temp_p->size);
    gsl_vector_free(temp_p);
    return err;
}


double compute_pme(const gsl_vector *p, const gsl_vector *t){
    double err = 0;
    gsl_vector *temp_p = gsl_vector_alloc(p->size);
    gsl_vector_memcpy(temp_p, p);
    gsl_vector_div (temp_p, t);
    gsl_vector_add_constant(temp_p, -1.0);
    err = gsl_blas_dasum (temp_p);
    err = err/(double)(temp_p->size);
    gsl_vector_free(temp_p);
    return err;
}


/* useful functions for sparse matrices computation */
// WARNING and NOTE: all below assume non-zero is 1; corresponding full vector is not used

//compute y = alpha * x + y
int gsl_blas_spdaxpy(const double alpha, const gsl_vector *x, const gsl_vector *xnz, gsl_vector* y){
    for (int i = 0; i< xnz->size; i++) {
        int changeind = (int)gsl_vector_get(xnz, i);
        double original = gsl_vector_get(y, changeind);
        double delta = alpha;
        gsl_vector_set(y, changeind, original + delta);
    }
    return 0;
}

//v1 is sparse, with nz elets 1
int gsl_spvector_scale(gsl_vector * v1, const gsl_vector * v1_1inds, double scalor){
    gsl_vector_set_zero(v1);
    for (int i = 0; i < v1_1inds->size; i++) {
        int changeind = (int)gsl_vector_get(v1_1inds, i);
        gsl_vector_set(v1, changeind, scalor);
    }
    return 0;
}

//v2 is a sp vector, v2_1inds is a vector storing indexes of non-zero elements in v2
int gsl_spvector_add(gsl_vector * v1, const gsl_vector * v2, const gsl_vector * v2_1inds){
    for (int i = 0; i < v2_1inds->size; i++) {
        int changeind = (int)gsl_vector_get(v2_1inds, i);
        gsl_vector_set(v1, changeind, gsl_vector_get(v1, changeind)+1);
    }
    return 0;
}

int gsl_spvector_sub(gsl_vector * v1, const gsl_vector * v2, const gsl_vector * v2_1inds){
    for (int i = 0; i < v2_1inds->size; i++) {
        int changeind = (int)gsl_vector_get(v2_1inds, i);
        gsl_vector_set(v1, changeind, gsl_vector_get(v1, changeind) - 1);
    }
    return 0;
}

//given two vectors of non-zero elements indexes
//this function returns a vector of all nonzero indexes
gsl_vector * gsl_spvector_get_nz2vecs(const gsl_vector * v1_1inds, const gsl_vector * v2_1inds){
    gsl_vector * nz_inds = gsl_vector_alloc(v1_1inds->size + v2_1inds->size);
    gsl_vector_view v1nz = gsl_vector_subvector(nz_inds, 0, v1_1inds->size);
    gsl_vector_memcpy(&v1nz.vector, v1_1inds);
    int count = v1_1inds->size;
    for (int i = 0; i < v2_1inds->size; i++) {
        int indv2 = (int)gsl_vector_get(v2_1inds, i);
        int indicator = 0;
        for (int j = 0; j < v1_1inds->size; j++) {
            int indv1 = (int)gsl_vector_get(v1_1inds, j);
            if (indv2 == indv1) {
                indicator = 1;
                break;
            }
        }
        if (indicator == 0) {
            gsl_vector_set(nz_inds, count, indv2);
            count++;
        }
    }
    //count will be the length of the final non-zero vector
    gsl_vector * nzs = gsl_vector_alloc(count);
    gsl_vector_view nzsview = gsl_vector_subvector(nz_inds,0,count);
    gsl_vector_memcpy(nzs, &nzsview.vector);
    gsl_vector_free(nz_inds);
    return nzs;
}

//compute pairwise product of vector v1 and sparse vector v2
int gsl_spvector_mul(gsl_vector * v1, const gsl_vector * v2, const gsl_vector * v2_1inds){
    for (int i = 0; i < v2_1inds->size; i++) {
        int changeind = (int)gsl_vector_get(v2_1inds, i);
        gsl_vector_set(v1, changeind, gsl_vector_get(v1, changeind));
    }
    return 0;
}

//compute dot product for v1, v2, v2 is a sparse vector
// WARNING: Only works for binary v2
int gsl_blas_spddot(const gsl_vector * v1, const gsl_vector * v2, const gsl_vector * v2_1inds, double *result){
    double sum = 0;
    for (int i = 0; i < v2_1inds->size; i++) {
       int changeind = (int)gsl_vector_get(v2_1inds, i);
       //sum += gsl_vector_get(v1, changeind) * gsl_vector_get(v2, changeind);
        sum += gsl_vector_get(v1, changeind);
    }
    *result = sum;
    return 0;
}

//compute A*v1 -> v2, where A is a matrix and v1 is a sparse vector
int gsl_blas_dgemspv(const gsl_matrix * A, const gsl_vector * v1, const gsl_vector * v1_1inds, gsl_vector * v2){
    double v2i = 0;
    for (int i = 0 ; i< A->size1; i++) {
        gsl_vector_const_view Arow = gsl_matrix_const_row (A, i);
        gsl_blas_spddot(&Arow.vector, v1, v1_1inds, &v2i);
        gsl_vector_set(v2, i, v2i);
    }
    return 0;
}

//compute v1*A -> v2, where A is a matrix and v1 is a sparse vector
int gsl_blas_dgespvm(const gsl_matrix * A, const gsl_vector * v1, const gsl_vector * v1_1inds, gsl_vector * v2){
    double v2i = 0;
    for (int i = 0 ; i< A->size2; i++) {
        gsl_vector_const_view Acol = gsl_matrix_const_column (A, i);
        gsl_blas_spddot(&Acol.vector, v1, v1_1inds, &v2i);
        gsl_vector_set(v2, i, v2i);
    }
    return 0;
}

//compute A*v1 -> v2, where A is a sparse matrix and v1 is a vector
int gsl_blas_dgespmv(const gsl_matrix * A, const gsl_matrix * A1ind, const gsl_vector * v1, gsl_vector * v2){
    double v2i = 0;
    // TODO: warning, Arow is not used
    gsl_vector_const_view Arow = gsl_matrix_const_row (A, 0);
    for (int i = 0 ; i< A1ind->size1; i++) {
       //gsl_vector_const_view Arow = gsl_matrix_const_row (A, i);
        gsl_vector_const_view A1indrow = gsl_matrix_const_row (A1ind, i);
        gsl_blas_spddot(v1, &Arow.vector, &A1indrow.vector, &v2i);
        gsl_vector_set(v2, i, v2i);
    }
    return 0;
}

//compute AB and store result in C, where A is a sparse matrix
int gsl_blas_dgespmm(const gsl_matrix * A, const gsl_matrix * A1ind, const gsl_matrix * B, gsl_matrix * C){
    
    gsl_vector_const_view rowA = gsl_matrix_const_row(A, 0);
    for (int i = 0; i < A1ind->size1; i++) {
        gsl_vector_const_view rowA1ind = gsl_matrix_const_row(A1ind, i);
        gsl_vector_view rowC = gsl_matrix_row(C, i);
        gsl_blas_dgespvm(B, &rowA.vector, &rowA1ind.vector, &rowC.vector);
    }
    return 0;
}

//compute outer product for x*y^T, x is a sparse vector
int gsl_outer_product_sp(gsl_matrix *A, gsl_vector *v1, const gsl_vector * v1_1inds, gsl_vector *v2){
    gsl_matrix_set_zero(A);
    if (A->size1 != v1->size || A->size2 != v2->size) {
        printf("Error: Vector-Matrix dimension does not match!");
        return 0;
    }
    gsl_vector *temprow = gsl_vector_calloc(A->size2);
    for (int i = 0; i<v1_1inds->size; i++) {
        gsl_vector_memcpy(temprow, v2);
        //gsl_vector_scale(temprow, 1);
        gsl_matrix_set_row(A, (int)gsl_vector_get(v1_1inds,i), temprow);
    }
    gsl_vector_free(temprow);
    return 0;
}

/* Init PHI matrix by one indexes*/
int InitPHI(gsl_matrix *phi, const gsl_matrix *oneinds){
    for (int i = 0; i < oneinds->size1; i++) {
        gsl_vector_view phirow = gsl_matrix_row(phi, i);
        gsl_vector_const_view onesrow = gsl_matrix_const_row(oneinds, i);
        for (int j = 0; j < oneinds->size2; j++) {
            gsl_vector_set(&phirow.vector, (int)gsl_vector_get(&onesrow.vector, j), 1.0);
        }
    }
    return 0;
}


/* Matrix QR decomposition */

//get the QA matrix without forming Q matrix

int gsl_linalg_QR_Qmat (const gsl_matrix * QR, const gsl_vector * tau, gsl_matrix * A){
    const size_t M = QR->size1;
    const size_t N = QR->size2;
    int gslmin = M>=N?N:M; 
    if (tau->size != gslmin)
    {
        GSL_ERROR ("size of tau must be MIN(M,N)", GSL_EBADLEN);
    }
    else if (A->size1 != M)
    {
        GSL_ERROR ("matrix must have M rows", GSL_EBADLEN);
    }
    else
    {
        size_t i;
        
        /* compute Q A */
        
        for (i = gslmin; i-- > 0;)
        {
            gsl_vector_const_view c = gsl_matrix_const_column (QR, i);
            gsl_vector_const_view h = gsl_vector_const_subvector (&(c.vector), i, M - i);
            gsl_matrix_view m = gsl_matrix_submatrix(A, i, 0, M - i, A->size2);
            double ti = gsl_vector_get (tau, i);
            gsl_linalg_householder_hm (ti, &(h.vector), &(m.matrix));
        }
        return GSL_SUCCESS;
    }
}

//retain the first k rows in R matrix
int gsl_linalg_QR_unpackR (const gsl_matrix * QR, const gsl_vector * tau, const int k, gsl_matrix * R){
    
    const size_t M = QR->size1;
    const size_t N = QR->size2;
    int gslmin = M>=N?N:M;
    if (tau->size != gslmin)
    {
        GSL_ERROR ("size of tau must be MIN(M,N)", GSL_EBADLEN);
    }
    else
    {
        size_t i, j;
        
        /*  Form the right triangular matrix R from a packed QR matrix */
        
        for (i = 0; i < k; i++)
        {
            for (j = 0; j < i && j < N; j++)
                gsl_matrix_set (R, i, j, 0.0);
            
            for (j = i; j < N; j++)
                gsl_matrix_set (R, i, j, gsl_matrix_get (QR, i, j));
        }
        return GSL_SUCCESS;
    }
}






