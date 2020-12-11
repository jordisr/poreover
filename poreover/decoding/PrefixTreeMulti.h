#ifndef BEAM_MULTI_HPP
#define BEAM_MULTI_HPP

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <functional>
#include <cmath>

#include "Log.h"
#include "PrefixTree.h"
#include "Beam.h"

#define NUM_READS 3
#define DEFAULT_VALUE -std::numeric_limits<double>::infinity()

// first step is getting consensus algorithm working with constant number of reads...
class BonitoNodeMulti : public Node<BonitoNodeMulti>{
public:
  static const int dim = NUM_READS;
  std::unordered_map<int, double> probability[dim];
  std::unordered_map<int, double> probability_gap[dim];
  std::unordered_map<int, double> probability_no_gap[dim];

  double max_prob[dim];
  int last_t[dim];
  int max_t[dim];

  BonitoNodeMulti(int s, BonitoNodeMulti* p) : Node<BonitoNodeMulti>(s, p) {
    for (int d=0; d<dim; ++d) {
      max_prob[d] = DEFAULT_VALUE;
      last_t[d] = 0;
      max_t[d] = 0;
    }
  }
  BonitoNodeMulti(int s) : Node<BonitoNodeMulti>(s) {
    for (int d=0; d<dim; ++d) {
      max_prob[d] = DEFAULT_VALUE;
      last_t[d] = 0;
      max_t[d] = 0;
    }
  }
  BonitoNodeMulti() : Node<BonitoNodeMulti>() {
    for (int d=0; d<dim; ++d) {
      max_prob[d] = DEFAULT_VALUE;
      last_t[d] = 0;
      max_t[d] = 0;
    }
  }

   double probability_at(int n, int t) const {
    if (probability[n].count(t) > 0) {
      return probability[n].at(t);
    } else {
      return DEFAULT_VALUE;
    }
  }

  double probability_gap_at(int n, int t) const {
   if (probability_gap[n].count(t) > 0) {
     return probability_gap[n].at(t);
   } else {
     return DEFAULT_VALUE;
   }
 }

   double probability_no_gap_at(int n, int t) const {
    if (probability_no_gap[n].count(t) > 0) {
      return probability_no_gap[n].at(t);
    } else {
      return DEFAULT_VALUE;
    }
  }

  double probability_at(int t) const {
    return probability_at(0, t) + probability_at(1, t);
  }

  double last_probability() const {
    double prob_sum = 0;
    for (int n=0; n<dim; ++n) {
      prob_sum += probability[n].at(last_t[n]);
    }
    return prob_sum;
  }

  double max_probability_sym() const {
    double prob_sum = 0;
    for (int d=0; d<dim; ++d) {
      prob_sum += max_prob[d];
    }
    return prob_sum;
  }

  double max_probability_sym_safe() const {
    double prob_sum = 0;
    for (int d=0; d<dim; ++d) {
      //if (! isinf(max_prob[d])) {
        prob_sum += max_prob[d];
      //}
    }
    return prob_sum;
  }

  double max_probability() const {
    return max_probability_sym_safe();
  }

  void reset_max() {
    for (int d=0; d<dim; ++d) {
      max_prob[d] = DEFAULT_VALUE;
    }
  }

  void reset_max(int i) {
    if (i < dim) {
      max_prob[i] = DEFAULT_VALUE;
    }
  }

  void set_probability(int i, int t, double gap_val, double no_gap_val) {
    probability[i][t] = logaddexp(gap_val, no_gap_val);
    probability_gap[i][t] = gap_val;
    probability_no_gap[i][t] = no_gap_val;
    last_t[i] = t;
    if (probability[i][t] > max_prob[i]) {
      max_prob[i] = probability[i][t];
      max_t[i] = t;
    }
  }

};

class BonitoPrefixTreeMulti : public PrefixTree<BonitoNodeMulti*> {
public:
  static const int dim = NUM_READS;
  int t_max[dim];
  double **y[dim];
  int gap_char;

  BonitoPrefixTreeMulti(double **d1, int u, double **d2, int v, std::string a) : PrefixTree<BonitoNodeMulti*>(a) {

    y[0] = d1;
    y[1] = d2;

    t_max[0] = u;
    t_max[1] = v;

    gap_char = alphabet.length();
    root = new BonitoNodeMulti(gap_char);
    for (int d=0; d<dim; ++d) {
      root->probability[d][-1] = 0;
      root->probability_gap[d][-1] = 0;
      root->probability_no_gap[d][-1] = DEFAULT_VALUE;
    }

  }

  BonitoPrefixTreeMulti(double **arg_y, int *arg_t_max, std::string a) : PrefixTree<BonitoNodeMulti*>(a) {

    int t_max_cum = 0;
    for (int d=0; d<dim; ++d) {
      y[d] = &arg_y[t_max_cum];
      t_max_cum += arg_t_max[d];
    }

    gap_char = alphabet.length();
    root = new BonitoNodeMulti(gap_char);
    for (int d=0; d<dim; ++d) {
      root->probability[d][-1] = 0;
      root->probability_gap[d][-1] = 0;
      root->probability_no_gap[d][-1] = DEFAULT_VALUE;
    }
  }

  void update_prob(BonitoNodeMulti* n, int i, int t) {
    double gap_prob = n->probability_at(i, t-1) + y[i][t][gap_char];
    double no_gap_prob;

    if (n->parent->depth == 0 && t==0) {
        no_gap_prob = y[i][t][n->last];
    } else if (n->parent->last == n->last) {
      no_gap_prob = logaddexp(n->parent->probability_gap_at(i,t-1) + y[i][t][n->last], n->probability_no_gap_at(i,t-1) + y[i][t][n->last]);
    } else {
      no_gap_prob = logaddexp(n->parent->probability_at(i,t-1) + y[i][t][n->last], n->probability_no_gap_at(i,t-1) + y[i][t][n->last]);
    }

    //std::cout << "i:" << i << "\tt:" << t << "\tgap_prob:" << gap_prob << "\tno_gap_prob:" << no_gap_prob << "\n";
    n->set_probability(i, t, gap_prob, no_gap_prob);

  }

};

int print_data(double **arg_y, int *arg_t_max, std::string alphabet) {
  static const int dim = NUM_READS;

  double **y[dim];
  int t_max_cum = 0;

  for (int d=0; d<dim; ++d) {
    y[d] = &arg_y[t_max_cum];
    t_max_cum += arg_t_max[d];
  }

  std::cout << "Writing data from print_data...\n";
  for (int d=0; d<dim; ++d) {
    std::cout << "Read " << d << "\n";
    int this_t_max = arg_t_max[d];
    for (int t=0; t<this_t_max; ++t) {
      std::cout << "\tRow " << t << "\t";
      for (int c=0; c<=alphabet.length(); ++c) {
        std::cout << "\t" << y[d][t][c];
      }
      std::cout << "\n";
    }
  }
  return 0;
}

std::string beam_polish(double **arg_y, int *arg_t_max, int **arg_envelope_ranges, std::string reference, std::string alphabet, int beam_width) {

    static const int dim = NUM_READS;
    double **y[dim];
    int **envelope_ranges[dim];

    // split concatenated logits and alignment envelope ranges
    int e = 0;
    int t_max_cum = 0;
    for (int d=0; d<dim; ++d) {
      y[d] = &arg_y[t_max_cum];
      t_max_cum += arg_t_max[d];
      envelope_ranges[d] = &arg_envelope_ranges[e];
      e += reference.length();
    }

    BonitoPrefixTreeMulti tree(arg_y, arg_t_max, alphabet);
    Beam<BonitoNodeMulti*, node_greater_max_lengthnorm<BonitoNodeMulti*>> beam_(beam_width);

    auto children = tree.expand(tree.root);

    // initialize beam
    for (int i=0; i<children.size(); i++) {
        //std::cout << "expanding child " << i << "\n";
        auto n = children[i];
        for (int d=0; d<dim; ++d) {
          tree.update_prob(n, d, 0);
        }
        beam_.push(n);
    }

    // map reference string to character indices
    // (currently don't use the reference sequence )
    /*
    int alphabet_size = alphabet.length();
    std::unordered_map<char, int> alphabet_map;
    for (int a=0; a<alphabet_size; a++) {
        alphabet_map[alphabet[a]] = a;
    }

    int *target_int = new int[reference.length()];
    for (int s=0; s<reference.length(); s++) {
        target_int[s] = alphabet_map[reference[s]];
    }
    */

    for (int t=0; t<reference.length(); ++t) {

      for (int b=0; b < beam_width; ++b) {

          auto beam_node = beam_.elements[b];
          beam_node->reset_max();

          for (int d=0; d<dim; d++) {
            //for (int td=0; td<arg_t_max[d]; td++) {
            int td_start = envelope_ranges[d][t][0];
            int td_end = envelope_ranges[d][t][1];
            for (int td=td_start; td<td_end; td++) {
              tree.update_prob(beam_node, d, td);
            }
          }

          // expand node and add children to the beam
          auto children = tree.expand(beam_node);
          /*
          for (int i=0; i<children.size(); i++) {
              auto child = children[i];
              for (int d=0; d<dim; d++) {
                //for (int td=0; td<arg_t_max[d]; td++) {
                int td_start = envelope_ranges[d][t][0];
                int td_end = envelope_ranges[d][t][1];
                //std::cout << "start:" << td_start << "\tend:" << td_end <<"\n";
                for (int td=td_start; td<td_end; td++) {
                  tree.update_prob(child, d, td);
                }
              }
              beam_.push(child);
          }
          */
          for (int i=0; i<children.size(); i++) {
              auto child = children[i];
              auto grandchildren = tree.expand(child);
              for (int d=0; d<dim; d++) {
                //for (int td=0; td<arg_t_max[d]; td++) {
                int td_start = envelope_ranges[d][t][0];
                int td_end = envelope_ranges[d][t][1];
                //std::cout << "start:" << td_start << "\tend:" << td_end <<"\n";
                for (int td=td_start; td<td_end; td++) {
                  tree.update_prob(child, d, td);
                }

                // expanding twice to allow for sequences longer than ref
                for (int j=0; j<grandchildren.size(); j++) {
                  auto grandchild = grandchildren[i];
                  for (int td=td_start; td<td_end; td++) {
                    tree.update_prob(grandchild, d, td);
                  }
                  beam_.push(grandchild);
                }

              }
              beam_.push(child);
          }

        }

        beam_.prune();
        //std::cout << "Beam after pruning\n";
        /*
        for (int b=0; b < beam_.size(); b++) {
            std::cout << "last_t=" << beam_.elements[b]->last_t[0] << "\n";
            std::cout << "t=" << t << "----" << tree.get_label(beam_.elements[b]) << " : " << beam_.elements[b]->max_probability() << "\tmax[1] at v=" << beam_.elements[b]->max_t[1] << "\n";
            std::cout << "t=" << t << "----" << tree.get_label(beam_.elements[b]) << " : " << beam_.elements[b]->max_prob[0] << "+" << beam_.elements[b]->max_prob[1] << "+" << beam_.elements[b]->max_prob[2] << "\n";
        }
        */

    }

    // just output statistics from top node
    auto top_node_ = beam_.top();
    std::string top_node_label = tree.get_label(top_node_);

    return top_node_label;

}

#endif
