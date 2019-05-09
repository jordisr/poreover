#ifndef PREFIX_TREE_HPP
#define PREFIX_TREE_HPP

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <functional>
#include <cmath>

#include <Log.h>

#define DEFAULT_VALUE -std::numeric_limits<double>::infinity()

class Node {
public:
  int last;
  Node* parent;
  std::unordered_map<int, double> probability;
  std::vector<Node*> children;
  int max_t = 0;

  Node(int s, Node* p) :last{s}, parent{p} {}
  Node(int s) :last{s}, parent{nullptr} {}
  Node() :last{-1}, parent{nullptr} {}

  int get_last() const { return last; };
  Node* get_parent() const { return parent; };

   double probability_at(int t) const {
    if (probability.count(t) > 0) {
      return probability.at(t);
    } else {
      return DEFAULT_VALUE;
    }
  }

  void set_probability(int t, double val) {
    probability[t] = val;
    max_t = t;
  };

  Node* add_child(int c) {
    Node* child = new Node(c);
    child->parent = this;
    children.push_back(child);
    return child;
  }

};

template <class T>
bool node_greater(T n1, T n2) {
  return (n1->probability_at(n1->max_t) > n2->probability_at(n2->max_t));
}

template <class TNode>
class PrefixTree {
public:
    double **y;
    int t_max;
    std::string alphabet;
    TNode root;

    PrefixTree(double **d, int v, std::string a) : y{d}, t_max{v}, alphabet{a} {}

    // expand children if node hasn't been expanded
    std::vector<TNode> expand(TNode n) {
      if (n->children.size() == 0) {
        for (int i=0; i < alphabet.length(); i++) {
          n->add_child(i);
        }
      }
      return n->children;
    }

    // trace path back to root and output label
    std::string get_label(TNode n) {
      std::string label = "";
      TNode prefix = n;
      while (prefix) {
        label = alphabet[prefix->last] + label;
        prefix = prefix->parent;
      }
      return label;
    }

};

class PoreOverPrefixTree : public PrefixTree<Node*> {
public:
  int gap_char;

  PoreOverPrefixTree(double **d, int v, std::string a) : PrefixTree<Node*>(d, v, a) {
    gap_char = alphabet.length();
    root = new Node(gap_char);
    root->probability[-1] = 0;
    double blank_sum = 0;
    for (int i=0; i<t_max; i++) {
      blank_sum += y[i][gap_char];
      root->probability[i] = blank_sum;
    }
  }

  void update_prob(Node* n, int t) {
    double a = n->parent->probability_at(t-1);
    double b = y[t][n->last];
    double emit_state = a+b;

    double c = n->probability_at(t-1);
    double d = y[t][gap_char];
    double stay_state = c+d;

    n->set_probability(t, logaddexp(emit_state, stay_state));
  };

};

#endif
