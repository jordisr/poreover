#ifndef FORWARD_HPP
#define FORWARD_HPP

#include <string>
#include <iostream>
#include <unordered_map>
#include <limits>
#include <cmath>
#include <algorithm>

#include "SparseMatrix.h"
#include "Log.h"

std::string viterbi_acceptor_poreover(double **y, int t_max, int band_size, std::string label, std::string alphabet) {

  int l_max = label.length();
  int alphabet_size = alphabet.length();
  int gap_char = alphabet_size;

  std::cout << "Mapping label\n";
  // map label string to character indices
  std::unordered_map<char, int> alphabet_map;
  for (int a=0; a<alphabet_size; a++) {
      alphabet_map[alphabet[a]] = a;
  }

  int *label_int = new int[l_max];
  for (int l=0; l<l_max; l++) {
      label_int[l] = alphabet_map[label[l]];
  }

  SparseMatrix<double> v;
  SparseMatrix<int> ptr(0);

  double emit_prob;
  double stay_prob;
  double gap_prob;

  v.push_row(0,band_size);
  v.push_row(0,band_size);
  ptr.push_row(0,band_size);
  ptr.push_row(0,band_size);

  gap_prob = 0;
  for (int t=0; t<t_max; t++) {
    gap_prob += y[t][gap_char];
    v.set(0, t, gap_prob);
    ptr.set(0, t, 0);
  }

  v.set(1,0,y[0][label_int[0]]);
  ptr.set(0,0,0);
  ptr.set(1,0,1);

  int row_start, row_end;

  for (int l=1; l<=l_max; l++) {

    row_start = std::max(1, int(l*(double)t_max/(double)l_max)-band_size);
    row_end = std::min(t_max, int(l*(double)t_max/(double)l_max)+band_size);

    //std::cout << l << "/" << l_max << "," << t_max << ":"<< row_start << ":" << row_end << "~" << l*(double)t_max/(double)l_max<< "\n";

    v.push_row(row_start, row_end);
    ptr.push_row(row_start, row_end);

    for (int t=row_start; t<row_end; t++) {
      if (t >= l-1) {
        emit_prob = y[t][label_int[l-1]] + v.get(l-1,t-1);
        stay_prob = y[t][gap_char] + v.get(l,t-1);
        if (emit_prob >= stay_prob) {
          v.set(l,t,emit_prob);
          ptr.set(l,t,1);
        } else {
          v.set(l,t,stay_prob);
          ptr.set(l,t,0);
        }
      }
    }

  }

 /*
  // print out matrix for debugging
  for (int l=0; l <= l_max; l++) {
      for (int t=0; t < t_max; t++) {
          std::cout << "v("<< l << "," << t << "):" << exp(v.get(l,t)) << "\t";
          //std::cout << v.get(l,t) << "\t";
      }
      std::cout << "\n";
      for (int t=0; t < t_max; t++) {
          std::cout << "ptr("<< l << "," << t << "):" << ptr.get(l,t) << "\t";
          //std::cout << v.get(l,t) << "\t";
      }
      std::cout << "\n\n";
  }
  */

  // traceback best path
  std::string path = "";
  for (int t=0; t<t_max; t++) {
      path += std::to_string(gap_char);
  }

  int l = l_max;
  int t = t_max-1;

  while (l>0) {
      //std::cout << "Traceback:" << l << "," << t << ":" << ptr.get(l,t) << "\n";
    if (ptr.get(l,t) > 0) {
        path[t] = std::to_string(label_int[l-1])[0];
      l -= 1;
    }
    t -= 1;
  }

  delete [] label_int;

  return path;

}

#endif
