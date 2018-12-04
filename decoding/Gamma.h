#ifndef GAMMA_H
#define GAMMA_H

#include "SparseMatrix.h"

double logaddexp(double, double);

double pair_gamma_log_envelope(double**, double**, int**, int, int, int);

void pair_gamma_log_envelope_inplace(SparseMatrix, SparseMatrix, double**, double**, int**, int, int, int);

#endif
