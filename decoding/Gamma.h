#ifndef GAMMA_H
#define GAMMA_H

#include "SparseMatrix.h"

//double pair_gamma_log_envelope(double[][3], double[][3], int[][2], int, int);
//double pair_gamma_log_envelope(double (*)[3], double (*)[3], int (*)[2], int, int);

double pair_gamma_log_envelope(double**, double**, int**, int, int, int);
//double pair_gamma_log_envelope(double*, double*, int*, int, int);

#endif
