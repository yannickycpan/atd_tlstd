This code base compares policy evaluation algorithms on random MDPs and trajectories, such as those from mountain car with a fixed policy. To compile the code, type

>>> make

To run the default setting (a random MDP with 30 states), with 20 runs and 2000 steps, type

>>>./compare_algorithms 20 2000 1

where the 1 specifies type. Type 1 is to use model MDP (generate data and learn from data interactively), currently it will use boyan chain. To modify it to RandomMDP, please modify the makefile. 

To plot the results, use:

>>> julia script_plot.jl(this may not work due to frequent revisions)

and the figure will be outputted into the specified figurename, in the results directory. See the short script_plot.jl to modify the settings. 


###############################
Code structure
###############################

The algorithms are specified in the algorithms folder. The structure is object-oriented. The file algorithm.c contains the parent class, with alg_t storing the required functions and variables for the algorithm class. The subclasses are in linear_algorithms and matrix_algorithms.c. Each algorithm (e.g. TD and TO-TD) could have had their own file, but it was redundant to do so bcause the linear algorithms are all quite similar. Instead, linear_algorithms is a subclass with constructor and destructor information for all of the linear algorithms. In experiment_utils.c, the experimenter just needs to know the name of the algorithm, the constructor and the destructor. This is specified in the algorithms_list struct in experiment_utils.h. 

To add new algorithms, either a new algorithm file can be created (e.g., sketching_algorithms.c) or it can be added to the algorithms currently in linear algorithms and matrix algorithms.

###############################
Parameter sweeps
###############################

The alg_params_t struct is currently generic, to enable a broad range of parameter sweeps. It is not necessary to sweep all the parameters in alg_params_t; one simply creates an array of alg_params_t that specifies the desired parameter combinations. This approach also allows different parameters to be swept for different algorithms. For example, if you want to sweep step-sizes for one algorithm and rank for another algorithm (which does not use step-sizes), then you can simply make the step-size and rank change together for each alg_params_t in the array. For example, if you want to try 2 step-sizes 0.1 and 1.0 and two ranks 5 and 10, then you would set

int num_params = 2;
struct alg_params_t alg_params[num_params];
alg_params[0].alpha = 0.1;
alg_params[0].rank = 5;
alg_params[1].alpha = 1.0;
alg_params[1].rank = 10;

#######update 10/20###########
this code is now runnable on the cluster
1. *.pbs file is the job script
2. some mergefile.m will merge the results and then store it under RMDPresults/. The files from jobs will be under RMDPresults/jobs/. It will generate a missid.txt if there are missing jobs. to use this script, type 'matlab -r "mergefile(number of parameter settings)"' from command line.
