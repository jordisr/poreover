#include <vector>
#include <iostream>
#include <limits>
#include <string>
#include <map>
#include <cmath>

#include "SparseMatrix.h"
#include "Gamma.h"

#define DEFAULT_VALUE -std::numeric_limits<double>::infinity()

using namespace std;

void print_row(double* row, int row_size) {
  for (int i=0; i<row_size; i++) {
    cout << row[i] << " ";
  }
  cout << endl;
}

void forward_vec_log(string alphabet, int s, int i, int t_max, double** y, double* fw, double* fw_prev) {
  int alphabet_size = alphabet.length();
  for (int t=0; t<t_max; t++) {
    if (t==0) {
      if (i==1) {
          fw[t] = y[t][s];
      }
    } else {
      fw[t] = logaddexp(y[t][alphabet_size]+fw[t-1], y[t][s]+fw_prev[t-1]);
    }
  }
}

void forward_vec_log(string alphabet, int s, int i, int t_max, double** y, double* fw) {
  int alphabet_size = alphabet.length();
  for (int t=0; t<t_max; t++) {
    if (t == 0) {
        fw[t] = y[t][s];
    } else {
      fw[t] = y[t][alphabet_size]+fw[t-1];
    }
  }
}

void forward_vec_no_gap_log(string alphabet, int s, int i, int t_max, double** y, double* fw, double* fw_prev) {
  int alphabet_size = alphabet.length();
  if (i==1) {
    fw[0] = y[0][s];
  } else {
    fw[0] = DEFAULT_VALUE;
  }
  for (int t=1; t<t_max; t++) {
      fw[t] = fw_prev[t-1]+y[t][s];
  }
}

void forward(string alphabet, string label, double** y, int U) {
  double* fw_prev = new double[U];
  double* fw = new double[U];
  int label_size = label.length();
  int alphabet_size = alphabet.length();
  forward_vec_log(alphabet, alphabet_size, 0, U, y, fw_prev);
  print_row(fw_prev, U);
  for (int s=0; s<label_size; s++){
    int label_i;
    for (int i=0; i<alphabet_size; i++) {
      if (label[s] == alphabet[i]) {
        label_i = i;
      }
    }
    forward_vec_log(alphabet, label_i, s+1, U, y, fw, fw_prev);
    print_row(fw, U);
    for (int u=0; u<U; u++) {
      fw_prev[u] = fw[u];
      fw[u] = DEFAULT_VALUE;
    }
  }
}

string pair_prefix_search_log(double **y1, double **y2, int **envelope_ranges, int U, int V, string alphabet) {

  // initialize prefix search variables
  int alphabet_size = alphabet.length();
  bool continue_search = true;
  int search_level = 0;
  string curr_label = "";

  // gamma matrix DP
  SparseMatrix gamma_, gamma_ast;
  for (int u=0; u<U+1; u++) {
    gamma_.push_row(envelope_ranges[u][0],envelope_ranges[u][1]);
    gamma_ast.push_row(envelope_ranges[u][0],envelope_ranges[u][1]);
  }
  pair_gamma_log_envelope_inplace(gamma_, gamma_ast, y1, y2, envelope_ranges, U, V, alphabet_size+1);
  //cout << "Gamma(0,0): " << gamma_.get(0,0) << endl;

  // get gap Probability
  double gap_prob = 0;
  for (int u=0; u<U; u++) {
    gap_prob += y1[u][alphabet_size];
  }
  for (int v=0; v<V; v++) {
    gap_prob += y2[v][alphabet_size];
  }

  // keeping track of top label and prefix
  string best_label = "";
  string best_label_prev = "";
  double best_label_prob = gap_prob;
  double best_label_prob_prev = gap_prob;

  // first row of each forward matrix
  double* alpha1_prev = new double[U];
  double* alpha2_prev = new double[V];
  fill_n(alpha1_prev, U, DEFAULT_VALUE);
  fill_n(alpha2_prev, V, DEFAULT_VALUE);

  forward_vec_log(alphabet, alphabet_size, 0, U, y1, alpha1_prev);
  forward_vec_log(alphabet, alphabet_size, 0, V, y2, alpha2_prev);

  // initialize forward vectors
  double* alpha_ast1 = new double[U];
  double* alpha_ast2 = new double[V];
  fill_n(alpha_ast1, U, DEFAULT_VALUE);
  fill_n(alpha_ast2, V, DEFAULT_VALUE);

  double** alpha1 = new double*[alphabet_size];
  double** alpha2 = new double*[alphabet_size];

  while (continue_search) {
    search_level++;

    int best_prefix_i = 9000;
    double best_prefix_prob = DEFAULT_VALUE;

    // iterate over each non-gap character in alphabet
    for (int i=0; i<alphabet_size; i++) {
      string prefix = curr_label + alphabet[i];

      // calculate prefix probability
      forward_vec_no_gap_log(alphabet, i, search_level, U, y1, alpha_ast1, alpha1_prev);
      forward_vec_no_gap_log(alphabet, i, search_level, V, y2, alpha_ast2, alpha2_prev);

      /*
      cout << "alpha1:";
      print_row(alpha1_prev,U);
      cout << "alpha2:";
      print_row(alpha2_prev,U);


      cout << "alpha_ast1:";
      print_row(alpha_ast1,U);
      cout << "alpha_ast2:";
      print_row(alpha_ast2,U);
      */

      double prefix_prob = DEFAULT_VALUE;
      for (int u=0; u<=U; u++) {
        int row_start = envelope_ranges[u][0];
        int row_end = envelope_ranges[u][1];
        for (int v=row_start; v<=row_end; v++) {
          //cout << prefix_prob << ':' << u << ',' << v << ':' << alpha_ast1[u] + alpha_ast2[v] << ',' << gamma_.get(u+1,v+1) << endl;
          prefix_prob = logaddexp(prefix_prob, alpha_ast1[u]+alpha_ast2[v]+gamma_.get(u+1,v+1));
        }
      }
      prefix_prob -= gamma_.get(0,0);

      if (prefix_prob > best_prefix_prob) {
        best_prefix_prob = prefix_prob;
        best_prefix_i = i;
      }

      // calculate label probability
      alpha1[i] = new double[U];
      alpha2[i] = new double[V];
      fill_n(alpha1[i], U, DEFAULT_VALUE);
      fill_n(alpha2[i], V, DEFAULT_VALUE);
      forward_vec_log(alphabet, i, search_level, U, y1, alpha1[i], alpha1_prev);
      forward_vec_log(alphabet, i, search_level, V, y2, alpha2[i], alpha2_prev);

      //cout <<  alpha1[i][U-1] + alpha2[i][V-1] << ":" <<  gamma_.get(0,0) << endl;
      double label_prob = alpha1[i][U-1] + alpha2[i][V-1] - gamma_.get(0,0);
      if (label_prob > best_label_prob_prev) {
        best_label_prob_prev = label_prob;
        best_label_prev = prefix;
      }

      //cout << "search_level:" << search_level << " extending by " << alphabet[i] << " best_label:" << best_label << " label:" << prefix << " label_prob:" << label_prob << " prefix_prob:" << prefix_prob << endl;
    }

    // just for testing with a few iterations
    if (search_level > U) {
      continue_search = false;
    }

    if (best_prefix_prob < best_label_prob) {
      continue_search = false;
    } else {
      //cout << "Best prefix was " << alphabet[best_prefix_i] << endl;
      curr_label += alphabet[best_prefix_i];

      best_label = best_label_prev;
      best_label_prob = best_label_prob_prev;

      for (int u=0; u<U; u++) {
        alpha1_prev[u] = alpha1[best_prefix_i][u];
        alpha1[best_prefix_i][u] = DEFAULT_VALUE;
      }
      for (int v=0; v<V; v++) {
        alpha2_prev[v] = alpha2[best_prefix_i][v];
        alpha2[best_prefix_i][v] = DEFAULT_VALUE;
      }
    }
  }
  best_label = best_label_prev;
  best_label_prob = best_label_prob_prev;
  //string result = best_label + ":" + to_string(best_label_prob);
  string result = best_label;
  //cout << result << endl;
  return(result);
}

int main() {

  double y1[4][3] = {
    {-0.223144, -2.30259, -2.30259},
    {-2.30259, -1.20397, -0.510826},
    {-0.356675, -1.60944, -2.30259},
    {-2.30259, -2.30259, -0.223144}
  };

  double y2[4][3] = {
    {-0.356675,-1.60944,-2.30259},
    {-1.60944,-1.20397,-0.693147},
    {-0.356675,-1.60944,-2.30259},
    {-2.99573,-2.99573,-0.105361}
  };

  int U = 4;
  int V = 4;

  int envelope_ranges[5][2] = {
    // envelope ranges is [U+1][2], wich each row referring to V+1 elements
    {0,3+1},
    {0,3+1},
    {0,3+1},
    {0,3+1},
    {0,3+1} // extra row
  };

  // convert 2d arrays to 2d pointers
  double **y1_ptr;
  double **y2_ptr;
  y1_ptr = new double*[U];
  y2_ptr = new double*[U];
  for(int i=0; i<U; i++) {
    y1_ptr[i] = new double[3];
    y2_ptr[i] = new double[3];
    for (int j=0; j<3; j++) {
      y1_ptr[i][j] = y1[i][j];
      y2_ptr[i][j] = y2[i][j];
    }
  }

  // same thing for envelope ranges
  int **envelope_ranges_ptr = new int*[5];
  for(int i=0; i<4+1; i++) {
    envelope_ranges_ptr[i] = new int[2];
    envelope_ranges_ptr[i][0] = envelope_ranges[i][0];
    envelope_ranges_ptr[i][1] = envelope_ranges[i][1];
  }

  // test gamma DP calculation
  cout << pair_gamma_log_envelope(y1_ptr, y2_ptr, envelope_ranges_ptr, U, V, 3) << endl;
  double* fw = new double[U];
  for (int t=0; t<U; t++) {
    fw[t] = DEFAULT_VALUE;
  }

  // test forward vec calculation
  forward_vec_log("AB", 2, 0, U, y1_ptr, fw);
  for (int t=0; t<U; t++) {
    cout << fw[t] << " ";
  }
  cout << endl;

  cout << logaddexp(DEFAULT_VALUE, -5) << endl;
  cout << logaddexp(DEFAULT_VALUE, DEFAULT_VALUE) << endl;

  // test forward matrix
  cout << "Forward matrix" << endl;
  forward("AB", "AA", y1_ptr, U);

  // prefix search
  cout << "Pair prefix search: " << pair_prefix_search_log(y1_ptr, y2_ptr, envelope_ranges_ptr, U, V, "AB") << endl;;

  return 0;
}
