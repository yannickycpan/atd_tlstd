*******How to compile?***********
Use makefile, simply type 'make'. NOTE that you may need to revise the compiler name and library path.


*******General Explanations***********
1. This framework only allows you to deal with episodic data, the dataset mcphidir1024fs20rand2k.zip is the one we used in ATD paper for mountain car experiment. For energy domain experiment, I do not upload the data yet due to its large size, please contact me if you want that. I have another repository to deal with that continuous task.  
2. This framework only uses tile coded feature (so you can see sparse computation is used), you need to specify number of non zeros. It is easy to revise to non-sparse computation by replacing those _sp functions. 
3. This framework can be used on HPC facility in a [per parameter setting per node style]. Example of job script is in atd.job script, of course you may use different job schedular so you need to modify it.
4. Two ways of updating SVD are included: one is to update SVD per sample; another way is to update SVD in a batch manner. 

*******Guides for ATD and tLSTD *********

A matrix is maintained by U L Sigma V R five matrices. 

Use ATD as an example:

int atd_2ndorder_sp(gsl_vector * w, gsl_vector* b, gsl_vector * z, gsl_matrix ** matu, gsl_vector ** vecs, gsl_matrix ** matv, gsl_matrix ** matl, gsl_matrix ** matr, const double lambda_t, const double reward, const double gamma_t,const double gamma_tp1, gsl_vector * x_t, gsl_vector* x_tp1, const gsl_vector * x_t_1s, const gsl_vector* x_tp1_1s, const double alpha_t, double beta_t, const double xi, int r, int change_r, int * updated_svd);

w: weight vector we want to learn
b: used to explain how ATD relates to LSTD and TD, it is not used. 
z: eligibility trace.
matu, matl, vecs, matv, matr: five matrices mentioned above.
lambda: bootstrapping parameter.
alpha_t: stepsize.
beta_t: regularizer.
xi: truncation parameter.
r: input rank.
change_r: simply the step counter. Sorry for the confusion incurred by the name. 
updated_svd: used to indicate whether the SVD is already updated. 
x_t_1s/x_tp1_1s: one indexes.

The rest variables are MDP related. Each step for RL algorithm, we process a triple (x, x’, reward), which corresponds to x_t, x_tp1, reward. Gamma is the discount factor.  