ifeq ($(USER),yangpan)
	INCLUDE = -I/opt/local/include
	LIBS = -L/N/soft/cle4/gsl/1.15/lib -lgsl -lgslcblas -lm
else
	INCLUDE = -I/usr/local/include
	LIBS = -L/usr/local/lib -lgsl -lgslcblas -lm
endif
OPTS = -Wall -pedantic -std=c99

all: 
	make clean
	make utils.o
	make domain_mdp.o
	make algorithms.o
	make comp_algo_domain

clean:
	rm -f *.o comp_algo_domain

utils.o: 
	cc $(OPTS) $(INCLUDE) -c utils.c

domain_mdp.o: utils.o
	cc $(OPTS) $(INCLUDE) -c domain_mdp.c

algorithms.o: utils.o domain_mdp.o
	cc $(OPTS) $(INCLUDE) -c algorithms.c

comp_algo_domain: utils.o domain_mdp.o algorithms.o
	cc $(OPTS) $(INCLUDE) -o comp_algo_domain comp_algo_domain.c algorithms.o utils.o domain_mdp.o $(LIBS)
