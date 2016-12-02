#ifndef _RL_INTERFACE_H
#define _RL_INTERFACE_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// All MDPs must fill in transition info
struct transition_info_t {
      double reward;
      gsl_vector * x_t;
      gsl_vector * x_tp1;
      double gamma_t;
      double gamma_tp1;
    
      // For off-policy methods, add importance ratio
      double rho_tm1;
      double rho_t;
};


#endif
