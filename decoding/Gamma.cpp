#include "SparseMatrix.h"
#include <iostream>
#include <vector>
#include <cmath>

//const int alphabet_size = 3;
const double LOG_1 = 0.0;

using namespace std;

double logaddexp(double x1, double x2) {
  if (x1 >= x2) {
    return(x1 + log(1 + exp(x2-x1)));
  } else {
    return(x2 + log(1 + exp(x1-x2)));
  }
}

//double pair_gamma_log_envelope(double y1[][3], double y2[][3], int envelope_ranges[][2], int U, int V) {
//double pair_gamma_log_envelope(double (*y1)[3], double (*y2)[3], int (*envelope_ranges)[2], int U, int V) {
double pair_gamma_log_envelope(double **y1, double **y2, int **envelope_ranges, int U, int V, int alphabet_size) {
//double pair_gamma_log_envelope(double *y1, double *y2, int *envelope_ranges, int U, int V) {

  //cout << "U: " << U << " V: " << V << endl;

    // intialization
    SparseMatrix gamma_;
    SparseMatrix gamma_ast;

    //cout << "CHECKPOINT: Initializing gamma matrix" << endl;

    for (int u=0; u<U+1; u++) {
      //cout << "ENVELOPE RANGE " << u << " " << envelope_ranges[u][0] << " to " << envelope_ranges[u][1] << endl;
      gamma_.push_row(envelope_ranges[u][0],envelope_ranges[u][1]);
      gamma_ast.push_row(envelope_ranges[u][0],envelope_ranges[u][1]);
    }

    //cout << "gamma(" << U << ',' << V << ")=" << gamma_.get(U,V) << endl;
    gamma_.set(U,V,LOG_1);
    gamma_ast.set(U,V,LOG_1);
    //cout << "gamma(" << U << ',' << V << ")=" << gamma_.get(U,V) << endl;

    for (int v=0; v<V; v++) {
      double log_sum = 0.;
      for (int v_e=v; v_e<V; v_e++) {
        log_sum += y2[v_e][alphabet_size-1];
      }
      gamma_.set(U,v,log_sum);
      //gamma_ast.set(U,v,log_sum);
      //cout << "gamma(" << U << "," << v << ")=" << gamma_.get(U,v) << endl;
    }

    for (int u=0; u<U; u++) {
      double log_sum = 0.;
      for (int u_e=u; u_e<U; u_e++) {
        log_sum += y1[u_e][alphabet_size-1];
      }
      gamma_.set(u,V,log_sum);
      //gamma_ast.set(u,V,log_sum);
      //cout << "gamma(" << u << "," << V << ")=" << gamma_.get(u,V) << endl;
    }

    //cout << "CHECKPOINT: Starting DP calculations" << endl;

    double gamma_eps, gamma_ast_eps, gamma_ast_ast;

    for (int u=U-1; u>=0; u--) {
      int row_start = envelope_ranges[u][0];
      int row_end = envelope_ranges[u][1]-1;
      for (int v=row_end; v>=row_start; v--) {

        //cout << "(" << u << ',' << v << ')' << endl;

        gamma_eps = gamma_.get(u+1,v) + y1[u][alphabet_size-1];
        gamma_ast_eps = gamma_ast.get(u,v+1) + y2[v][alphabet_size-1];

        //cout << "gamma_eps=" << gamma_eps << endl;
        //cout << "gamma_ast_eps=" << gamma_ast_eps << endl;

        // logsumexp
        double total2 = 0.;
        for (int t=0; t<alphabet_size-1; t++) {
          total2 += exp(y1[u][t]+y2[v][t]);
        }
        gamma_ast_ast = gamma_.get(u+1,v+1) + log(total2);

        // storing DP matrices
        double logaddexp_;
        //logaddexp_ = log(exp(gamma_ast_eps) + exp(gamma_ast_ast));
        logaddexp_ = logaddexp(gamma_ast_eps, gamma_ast_ast);
        //cout << logaddexp_ << "/" << logaddexp(gamma_ast_eps, gamma_ast_ast) << endl;
        gamma_ast.set(u,v,logaddexp_);
        //logaddexp_ = log(exp(gamma_eps) + exp(gamma_ast.get(u,v)));
        logaddexp_ = logaddexp(gamma_eps, gamma_ast.get(u,v));
        gamma_.set(u,v,logaddexp_);

        //cout << "gamma(" << u << ',' << v << ")=" << gamma_.get(u,v) << endl;
      }
    }

  //cout << "CHECKPOINT: Finished DP calculations" << endl;

  return(gamma_.get(0,0));
}
