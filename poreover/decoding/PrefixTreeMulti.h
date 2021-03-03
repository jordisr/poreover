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

#define DEFAULT_VALUE -std::numeric_limits<double>::infinity()

class BonitoNodeMulti {
public:

  // number of reads
  const int dim;

  // general node variables
  int last;
  BonitoNodeMulti* parent;
  std::vector<BonitoNodeMulti*> children;
  int depth = 0;
  double beam_score;

  // for storing forward probabilities
  std::unordered_map<int, double>* probability;
  std::unordered_map<int, double>* probability_gap;
  std::unordered_map<int, double>* probability_no_gap;
  double* max_prob;
  int* last_t;
  int* max_t;

    BonitoNodeMulti(int d, int s, BonitoNodeMulti* p) :dim{d}, last{s}, parent{p} {
      probability = new std::unordered_map<int, double>[dim];
      probability_gap = new std::unordered_map<int, double>[dim];
      probability_no_gap = new std::unordered_map<int, double>[dim];

      max_prob = new double[dim];
      last_t = new int[dim];
      max_t = new int[dim];

      for (int i=0; i<dim; ++i) {

        probability[i][-1] = DEFAULT_VALUE;
        probability_gap[i][-1] = DEFAULT_VALUE;
        probability_no_gap[i][-1] = DEFAULT_VALUE;

        max_prob[i] = DEFAULT_VALUE;
        last_t[i] = 0;
        max_t[i] = 0;

      }
    }

    BonitoNodeMulti(int d, int s) : BonitoNodeMulti(d, s, nullptr) {}

    BonitoNodeMulti(int d) : BonitoNodeMulti(d, -1, nullptr) {}

    int get_last() const { return last; }

    BonitoNodeMulti* get_parent() const { return parent; }

    BonitoNodeMulti* add_child(int c) {
      BonitoNodeMulti* child = new BonitoNodeMulti(dim, c);
      child->parent = static_cast<BonitoNodeMulti*>(this);
      child->depth = this->depth + 1;
      children.push_back(child);
      return child;
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
        prob_sum += max_prob[d];
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

  void clear_on(int i) {
    if (depth <= i) {
      for (int n=0; n<dim; ++n) {
        probability[n].clear();
        probability_gap[n].clear();
        probability_no_gap[n].clear();
      }
    }
    for (auto x : children) {
        x->clear_on(i);
    }
  }

  virtual ~BonitoNodeMulti() {
      for (auto x : children) {
          delete x;
      }
      delete[] probability;
      delete[] probability_gap;
      delete[] probability_no_gap;
      delete[] max_prob;
      delete[] last_t;
      delete[] max_t;
  }

};

class BonitoPrefixTreeMulti : public PrefixTree<BonitoNodeMulti*> {
public:
  int dim;
  int* t_max;
  double ***y;
  int gap_char;

  BonitoPrefixTreeMulti(int dim_, double **arg_y, int *arg_t_max, std::string a) : PrefixTree<BonitoNodeMulti*>(a) {

    dim = dim_;
    t_max = new int[dim];
    y = new double**[dim];

    int t_max_cum = 0;
    for (int d=0; d<dim; ++d) {
      y[d] = &arg_y[t_max_cum];
      t_max_cum += arg_t_max[d];
    }

    gap_char = alphabet.length();
    root = new BonitoNodeMulti(dim, gap_char);
    for (int d=0; d<dim; ++d) {
      root->probability[d][-1] = 0;
      root->probability_gap[d][-1]  = 0;
      root->probability_no_gap[d][-1]  = DEFAULT_VALUE;
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

    n->set_probability(i, t, gap_prob, no_gap_prob);

  }

  virtual ~BonitoPrefixTreeMulti() {
    delete[] t_max;
    delete[] y;
  }

};

std::string beam_polish(int dim, double **arg_y, int *arg_t_max, int **arg_envelope_ranges, std::string reference, std::string alphabet, int beam_width, bool verbose, bool length_norm) {

    double*** y = new double**[dim];
    int*** envelope_ranges = new int**[dim];

    // split concatenated logits and alignment envelope ranges
    int e = 0;
    int t_max_cum = 0;
    for (int d=0; d<dim; ++d) {
      y[d] = &arg_y[t_max_cum];
      t_max_cum += arg_t_max[d];
      envelope_ranges[d] = &arg_envelope_ranges[e];
      e += reference.length();
    }

    BonitoPrefixTreeMulti tree(dim, arg_y, arg_t_max, alphabet);
    Beam<BonitoNodeMulti*, node_greater_beam<BonitoNodeMulti*>> beam_(beam_width);

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

            int td_start = envelope_ranges[d][t][0];
            int td_end = envelope_ranges[d][t][1];
            for (int td=td_start; td<td_end; td++) {
              tree.update_prob(beam_node, d, td);
            }
          }

          // expand node and add children to the beam
          auto children = tree.expand(beam_node);

          for (int i=0; i<children.size(); i++) {
              auto child = children[i];
              auto grandchildren = tree.expand(child);
              for (int d=0; d<dim; d++) {

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

        // calculate beam scores
        int ref_length = reference.length();

        for (int b=0; b < beam_.size(); b++) {
          if (length_norm) {
            // simple length normalization scheme
            beam_.elements[b]->beam_score = beam_.elements[b]->max_probability() / beam_.elements[b]->depth;
          } else {
            // no length normalization
            beam_.elements[b]->beam_score = beam_.elements[b]->max_probability();
          }
        }

        beam_.prune();
        if (verbose) {
          std::cout << "Beam after pruning\n";
          for (int b=0; b < beam_.size(); b++) {
              std::cout << "last_t=" << beam_.elements[b]->last_t[0] << "\n";
              std::cout << "t=" << t << "----" << tree.get_label(beam_.elements[b]) << " : " << beam_.elements[b]->max_probability() << "\n";
          }
        }

        if (t % 1000 == 0) {
          // TODO more efficient pruning method to reduce memory usage
          int min_depth = ref_length*2;
          for (int b=0; b < beam_.size(); b++) {
              if (beam_.elements[b]->depth < min_depth) {
                min_depth = beam_.elements[b]->depth;
              }
            }
          tree.root->clear_on(min_depth-2);
        }
    }

    // just output statistics from top node
    auto top_node_ = beam_.top();
    std::string top_node_label = tree.get_label(top_node_);

    delete[] y;
    delete[] envelope_ranges;

    return top_node_label;

}

#endif
