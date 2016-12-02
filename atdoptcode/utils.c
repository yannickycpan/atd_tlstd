
#include "utils.h"

/******************** Basic general-purpose functions ************/

/* string concat */
char* concat(char *s1, char *s2)
{
      char *result = malloc(strlen(s1)+strlen(s2)+1);//+1 for the zero-terminator
      //in real code you would check for errors in malloc here
      strcpy(result, s1);
      strcat(result, s2);
      return result;
}


/* 
 * For example, takes index (a,b,c) and flattens to single index 
 * index = a*n2*n3 + b*n3 + c */
int flatten_index(int * indices, const int * indexrange, const int numindices) {
      int index = 0;
      int i;
      int product = 1;
      for (i = numindices-1; i >= 0; i--) {
            index += indices[i]*product;
            product *= indexrange[i];
      }
      return index;
}

double gettime(struct timeval * tim) {
      gettimeofday(tim, NULL);
      return tim->tv_sec+(tim->tv_usec/1000000.0);    
}

int int2Binary(gsl_vector * array, int inv, int n)
{
      char* buffer;
      gsl_vector_set_all(array,0);
      int nbits = floor(log2(n))+1;
      if(inv == 0){
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

char* itoa(int val, int base){

      static char buf[32] = {0};

      int i = 30;

      for(; val && i ; --i, val /= base)

            buf[i] = "0123456789abcdef"[val % base];

      return &buf[i+1];

}

double log2(double a)
{
      return log(a)/log(2);
}

double min(double a, double b){
      if(a <= b)
            return a;
      return b; 
}

// Aggressive file open: if cannot open file, exits
// For all our settings, we cannot run the experiment unless
// We can open the required trajectory files
FILE * open_file(const char * filename, const char * open_argument) {
      FILE * file = fopen(filename, open_argument);
      if (file == NULL) {
            fprintf(stderr, "Could not open file %s\n", filename);
            exit(-1);
      }
      return file;
}


/******************** Functions for random generation ************/

void free_rgen(struct rgen_t * rgen) {
      gsl_rng_free(rgen->r);
}

void init_rgen(struct rgen_t * rgen, const unsigned long int seed) {
      const gsl_rng_type * T;

      gsl_rng_env_setup ();

      T = gsl_rng_default;
      rgen->r = gsl_rng_alloc (T);
      gsl_rng_set(rgen->r, seed);
}

double rand_un()
{
    return (double)rand() / (double)RAND_MAX;
}


double gaussian_random(struct rgen_t * rgen, double sigma, double mean) {
      return gsl_ran_gaussian(rgen->r,sigma) + mean;
}



int generate_random_indices(gsl_vector * v, const int n, struct rgen_t * rgen) {

      gsl_vector_set_all (v,-1);
      for (int i=0; i<v->size; i++){
            int val = uniform_random_int_in_range(rgen, n);
            while(gsl_vector_contains(v,val))
                  val = uniform_random_int_in_range(rgen, n);
            gsl_vector_set(v, i, val);
      }
      return 0;
}

int generate_random_uniform_matrix(gsl_matrix * mat, struct rgen_t * rgen, double min_v) {
      gsl_matrix_set_zero (mat);

      double v;
      for (int i=0; i<mat->size1; i++) {
            for (int j=0; j<mat->size2; j++){
                  v =  uniform_random(rgen);
                  if(v < min_v)
                        v = min_v;
                  gsl_matrix_set (mat, i, j,v);
            }
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

//the name correct???
int generate_woodruff_matrix(gsl_matrix * mat, struct rgen_t * rgen)
{
   gsl_matrix_set_zero (mat);

   for (int j=0; j<mat->size1; j++){
      gsl_vector_view rowview = gsl_matrix_row(mat, j);
      int setind = gsl_rng_uniform_int(rgen->r,(int)mat->size2);
      double setnum = rand_un()>0.5?1:-1;
      gsl_vector_set (&rowview.vector, setind, setnum);
   }
  
   return 0;
}

int generate_osnap_matrix(gsl_matrix * mat, struct rgen_t * rgen)
{
   gsl_matrix_set_zero (mat);
   int s = 150;
   double dest[150], src[10000];
   for(int i = 0; i<(int)mat->size2; i++)src[i] = i;
   gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
   gsl_ran_choose (r, dest, s, src, (int)mat->size2, sizeof (double));

   for (int j=0; j<mat->size1; j++){
      gsl_vector_view rowview = gsl_matrix_row(mat, j);
      for(int numnz = 0; numnz < s; numnz++){
        int setind = dest[numnz];
        double setnum = rand_un()>0.5?1/sqrt(s):-1/sqrt(s);
        gsl_vector_set (&rowview.vector, setind, setnum);
      }
   }
   gsl_rng_free(r);
   return 0;
}

//b is the batch size used to do aggregation
int generate_aggregate_matrix(gsl_matrix * mat, int b)
{
  gsl_matrix_set_zero (mat);
  double scalor = 1.0/b;
  int offset = 0;
  for (int i=0; i<mat->size2; i++) {
    gsl_vector_view col = gsl_matrix_column(mat, i);
    for (int j = offset; j< offset + b; j++){
      gsl_vector_set(&col.vector, j, scalor);
    }
    offset += b;
  }
  return 0;
}
// Assumes that the given probabilities are the cumulative sum
int sample_from_cdf(const double * cdf, struct rgen_t * rgen){

      double rand_n = uniform_random(rgen);
      int ind=0;
      while(cdf[ind]< rand_n)
            ind=ind+1;

      return ind;
}

double uniform_random(struct rgen_t * rgen) {
      return gsl_rng_uniform(rgen->r);
}

int uniform_random_int_in_range(struct rgen_t * rgen, const int n) {
      return gsl_rng_uniform_int(rgen->r,n);
}

/******************** Vector and matrix functions ************/

int gsl_matrix_print(gsl_matrix * m)
{
      for(int i=0;i<m->size1;i++){
            gsl_vector_view v = gsl_matrix_row(m,i);
            gsl_vector_print(&v.vector);
      }
      return 0;
}


/* This function is clearly unnecessary, but is much easier to read: c = A b */
int gsl_matrix_vector_product(gsl_vector *c, gsl_matrix *A, gsl_vector *b){
      gsl_blas_dgemv (CblasNoTrans, 1.0, A, b, 0, c);

      return 0;
}

int gsl_vector_contains(gsl_vector * a, const int j){
      for (int i =0; i < a->size; i++) {
            if(gsl_vector_get(a,i) == j)
                  return 1;
      }
      return 0;
}

int gsl_vector_print(gsl_vector * v)
{
      for(int i=0;i<v->size;i++)
            printf("%f ",gsl_vector_get(v,i));
      printf("\n");
      return 0;
}


int normalize_rows_matrix(gsl_matrix * mat) {
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

int normalize_vector(gsl_vector * vec) {
      double sum=0;
      for (int i=0; i<vec->size; i++) {
            sum = sum + gsl_vector_get (vec, i);
      }
      gsl_vector_scale(vec, 1.0/sum);

      return 0;
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

/* Init PHI matrix by one indexes*/
int InitPHI(gsl_matrix *phi, const gsl_matrix *oneinds){
    gsl_matrix_set_zero(phi);
    for (int i = 0; i < oneinds->size1; i++) {
        gsl_vector_view phirow = gsl_matrix_row(phi, i);
        gsl_vector_const_view onesrow = gsl_matrix_const_row(oneinds, i);
        for (int j = 0; j < oneinds->size2; j++) {
            int changeind = (int)gsl_vector_get(&onesrow.vector, j);
            gsl_vector_set(&phirow.vector, changeind, gsl_vector_get(&phirow.vector, changeind) + 1.0);
        }
    }
    return 0;
}

/* Sparse computation */
int gsl_blas_spddot(const gsl_vector * v1, const gsl_vector * v2_1inds, double *result){
    double sum = 0;
    for (int i = 0; i < v2_1inds->size; i++) {
       int changeind = (int)gsl_vector_get(v2_1inds, i);
        sum += gsl_vector_get(v1, changeind);
    }
    *result = sum;
    return 0;
}

int gsl_blas_dgespmv(gsl_matrix * A1ind, const gsl_vector * v1, gsl_vector * v2){
    double v2i = 0;
    for (int i = 0 ; i< A1ind->size1; i++) {
        gsl_vector_const_view A1indrow = gsl_matrix_const_row (A1ind, i);
        gsl_blas_spddot(v1, &A1indrow.vector, &v2i);
        gsl_vector_set(v2, i, v2i);
    }
    return 0;
}
