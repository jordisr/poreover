#ifndef GAMMA_H
#define GAMMA_H

#include "SparseMatrix.h"
#include "Log.h"
#include <vector>
#include <cmath>

#define DEFAULT_VALUE -std::numeric_limits<double>::infinity()

const double LOG_1 = 0.0;

//double pair_gamma_log_envelope(double y1[][3], double y2[][3], int envelope_ranges[][2], int U, int V) {
//double pair_gamma_log_envelope(double (*y1)[3], double (*y2)[3], int (*envelope_ranges)[2], int U, int V) {
double pair_gamma_log_envelope(double **y1, double **y2, int **envelope_ranges, int U, int V, int alphabet_size) {
//double pair_gamma_log_envelope(double *y1, double *y2, int *envelope_ranges, int U, int V) {

  //std::cout << "U: " << U << " V: " << V << endl;

    // intialization
    SparseMatrix<double> gamma_;
    SparseMatrix<double> gamma_ast;

    //std::cout << "CHECKPOINT: Initializing gamma matrix" << endl;

    for (int u=0; u<U+1; u++) {
      //std::cout << "ENVELOPE RANGE " << u << " " << envelope_ranges[u][0] << " to " << envelope_ranges[u][1] << endl;
      gamma_.push_row(envelope_ranges[u][0],envelope_ranges[u][1]);
      gamma_ast.push_row(envelope_ranges[u][0],envelope_ranges[u][1]);
    }

    //std::cout << "gamma(" << U << ',' << V << ")=" << gamma_.get(U,V) << endl;
    gamma_.set(U,V,LOG_1);
    gamma_ast.set(U,V,LOG_1);
    //std::cout << "gamma(" << U << ',' << V << ")=" << gamma_.get(U,V) << endl;

    for (int v=0; v<V; v++) {
      double log_sum = 0.;
      for (int v_e=v; v_e<V; v_e++) {
        log_sum += y2[v_e][alphabet_size-1];
      }
      gamma_.set(U,v,log_sum);
      //gamma_ast.set(U,v,log_sum);
      //std::cout << "gamma(" << U << "," << v << ")=" << gamma_.get(U,v) << endl;
    }

    for (int u=0; u<U; u++) {
      double log_sum = 0.;
      for (int u_e=u; u_e<U; u_e++) {
        log_sum += y1[u_e][alphabet_size-1];
      }
      gamma_.set(u,V,log_sum);
      //gamma_ast.set(u,V,log_sum);
      //std::cout << "gamma(" << u << "," << V << ")=" << gamma_.get(u,V) << endl;
    }

    //std::cout << "CHECKPOINT: Starting DP calculations" << endl;

    double gamma_eps, gamma_ast_eps, gamma_ast_ast;

    for (int u=U-1; u>=0; u--) {
      int row_start = envelope_ranges[u][0];
      int row_end = envelope_ranges[u][1]-1;
      for (int v=row_end; v>=row_start; v--) {

        //std::cout << "(" << u << ',' << v << ')' << endl;

        gamma_eps = gamma_.get(u+1,v) + y1[u][alphabet_size-1];
        gamma_ast_eps = gamma_ast.get(u,v+1) + y2[v][alphabet_size-1];

        //std::cout << "gamma_eps=" << gamma_eps << endl;
        //std::cout << "gamma_ast_eps=" << gamma_ast_eps << endl;

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
        //std::cout << logaddexp_ << "/" << logaddexp(gamma_ast_eps, gamma_ast_ast) << endl;
        gamma_ast.set(u,v,logaddexp_);
        //logaddexp_ = log(exp(gamma_eps) + exp(gamma_ast.get(u,v)));
        logaddexp_ = logaddexp(gamma_eps, gamma_ast.get(u,v));
        gamma_.set(u,v,logaddexp_);

        //std::cout << "gamma(" << u << ',' << v << ")=" << gamma_.get(u,v) << endl;
      }
    }

  //std::cout << "CHECKPOINT: Finished DP calculations" << endl;

  return(gamma_.get(0,0));
}

void pair_gamma_log_envelope_inplace(SparseMatrix<double> gamma_, SparseMatrix<double> gamma_ast, double **y1, double **y2, int **envelope_ranges, int U, int V, int alphabet_size) {
    gamma_.set(U,V,LOG_1);
    gamma_ast.set(U,V,LOG_1);

    for (int v=0; v<V; v++) {
      double log_sum = 0.;
      for (int v_e=v; v_e<V; v_e++) {
        log_sum += y2[v_e][alphabet_size-1];
      }
      gamma_.set(U,v,log_sum);
      //std::cout << "(" << U << "," << v << "):" << gamma_.get(U,v) << endl;
    }
    //std::cout << "Initialized v=0 to v=" << V << endl;

    for (int u=0; u<U; u++) {
      double log_sum = 0.;
      for (int u_e=u; u_e<U; u_e++) {
        log_sum += y1[u_e][alphabet_size-1];
      }
      gamma_.set(u,V,log_sum);
      //std::cout << "(" << u << "," << V << "):" << gamma_.get(u,V) << endl;
    }

    //std::cout << "gamma(U,V)=" << gamma_.get(U,V) << endl;
    //std::cout << "gamma(U+1,V+1)=" << gamma_.get(U+1,V+1) << endl;
    //std::cout << "gamma(U-1,V)=" << gamma_.get(U-1,V) << endl;
    //std::cout << "gamma(U,V-1)=" << gamma_.get(U,V-1) << endl;

    double gamma_eps, gamma_ast_eps, gamma_ast_ast;

    for (int u=U-1; u>=0; u--) {
      int row_start = envelope_ranges[u][0];
      int row_end = envelope_ranges[u][1]-1;
      for (int v=row_end; v>=row_start; v--) {
        //std::cout << "\t\t... v=" << v << endl;
        //std::cout << "\t\t... gamma(u+1,v)=" << gamma_.get(u+1,v) << endl;
        //std::cout << "\t\t... gamma(u,v+1)=" << gamma_.get(u,v+1) << endl;
        gamma_eps = gamma_.get(u+1,v) + y1[u][alphabet_size-1];
        gamma_ast_eps = gamma_ast.get(u,v+1) + y2[v][alphabet_size-1];
        //std::cout << "\t\t\t gamma_eps=" << gamma_eps << " gamma_ast_eps=" << gamma_ast_eps << endl;

        // logsumexp
        double total2 = 0.;
        for (int t=0; t<alphabet_size-1; t++) {
          total2 += exp(y1[u][t]+y2[v][t]);
        }
        gamma_ast_ast = gamma_.get(u+1,v+1) + log(total2);
        //std::cout << "\t\t\t gamma_ast_ast=" << gamma_ast_ast << endl;

        // storing DP matrices
        double logaddexp_;
        logaddexp_ = logaddexp(gamma_ast_eps, gamma_ast_ast);
        gamma_ast.set(u,v,logaddexp_);
        logaddexp_ = logaddexp(gamma_eps, gamma_ast.get(u,v));
        gamma_.set(u,v,logaddexp_);
      }
    }
}


#endif
