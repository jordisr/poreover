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

#include "Log.h"

#define DEFAULT_VALUE -std::numeric_limits<double>::infinity()

template <class N>
class Node {
public:
  int last;
  N* parent;
  std::vector<N*> children;
  int depth = 0;

  Node(int s, N* p) :last{s}, parent{p} {}
  Node(int s) :last{s}, parent{nullptr} {}
  Node() :last{-1}, parent{nullptr} {}
  virtual ~Node() {
      for (auto x : children) {
          delete x;
      }
  }

  int get_last() const { return last; }
  N* get_parent() const { return parent; }

  N* add_child(int c) {
    N* child = new N(c);
    child->parent = static_cast<N*>(this);
    child->depth = this->depth + 1;
    children.push_back(child);
    return child;
  }
};

class PoreOverNode : public Node<PoreOverNode> {
public:
  std::unordered_map<int, double> probability;
  int max_t = 0;

  PoreOverNode(int s, PoreOverNode* p) : Node<PoreOverNode>(s, p) {}
  PoreOverNode(int s) : Node<PoreOverNode>(s) {}
  PoreOverNode() : Node<PoreOverNode>() {}

  double probability_at(int t) const {
   if (probability.count(t) > 0) {
     return probability.at(t);
   } else {
     return DEFAULT_VALUE;
   }
 }

 double last_probability() const {
   return probability.at(max_t);
 }

 void set_probability(int t, double val) {
   probability[t] = val;
   max_t = t;
 }

};

class PoreOverNode2D : public Node<PoreOverNode2D> {
public:
  static const int dim = 2;
  std::unordered_map<int, double> probability[dim];
  int last_t[dim] = {0, 0};
  double last_prob[dim] = {0, 0};

  PoreOverNode2D(int s, PoreOverNode2D* p) : Node<PoreOverNode2D>(s, p) {}
  PoreOverNode2D(int s) : Node<PoreOverNode2D>(s) {}
  PoreOverNode2D() : Node<PoreOverNode2D>() {}

  double probability_at(int n, int t) const {
   if (probability[n].count(t) > 0) {
     return probability[n].at(t);
   } else {
     return DEFAULT_VALUE;
   }
 }

 double probability_at(int t) const {
    return probability_at(0, t) + probability_at(1, t);
}

double joint_probability(int u, int v) const {
   return probability_at(0, u) + probability_at(1, v);
}

 double last_probability() const {
   return last_prob[0] + last_prob[1];
 }

 void set_probability(int i, int t, double val) {
   probability[i][t] = val;
   last_t[i] = t;
   last_prob[i] = val;
 }

 void set_probability(int i, int t) {
   last_t[i] = t;
   last_prob[i] = probability_at(i, t);
 }

};


class FlipFlopNode : public Node<FlipFlopNode>{
public:
  std::unordered_map<int, double> probability;
  std::unordered_map<int, double> probability_flip;
  std::unordered_map<int, double> probability_flop;
  int max_t = 0;

  FlipFlopNode(int s, FlipFlopNode* p) : Node<FlipFlopNode>(s, p) {}
  FlipFlopNode(int s) : Node<FlipFlopNode>(s) {}
  FlipFlopNode() : Node<FlipFlopNode>() {}

   double probability_at(int t) const {
    if (probability.count(t) > 0) {
      return probability.at(t);
    } else {
      return DEFAULT_VALUE;
    }
  }

  double probability_flip_at(int t) const {
   if (probability_flip.count(t) > 0) {
     return probability_flip.at(t);
   } else {
     return DEFAULT_VALUE;
   }
 }

   double probability_flop_at(int t) const {
    if (probability_flop.count(t) > 0) {
      return probability_flop.at(t);
    } else {
      return DEFAULT_VALUE;
    }
  }

  double last_probability() const {
    return probability.at(max_t);
  }

  void set_probability(int t, double flip_val, double flop_val) {
    probability[t] = logaddexp(flip_val, flop_val);
    probability_flip[t] = flip_val;
    probability_flop[t] = flop_val;
    max_t = t;
  }

};

class FlipFlopNode2D : public Node<FlipFlopNode2D> {
public:
  static const int dim = 2;
  std::unordered_map<int, double> probability[dim];
  std::unordered_map<int, double> probability_flip[dim];
  std::unordered_map<int, double> probability_flop[dim];
  int last_t[dim] = {0, 0};

  FlipFlopNode2D(int s, FlipFlopNode2D* p) : Node<FlipFlopNode2D>(s, p) {}
  FlipFlopNode2D(int s) : Node<FlipFlopNode2D>(s) {}
  FlipFlopNode2D() : Node<FlipFlopNode2D>() {}

  double probability_at(int n, int t) const {
   if (probability[n].count(t) > 0) {
     return probability[n].at(t);
   } else {
     return DEFAULT_VALUE;
   }
  }

 double probability_flip_at(int n, int t) const {
  if (probability_flip[n].count(t) > 0) {
    return probability_flip[n].at(t);
  } else {
    return DEFAULT_VALUE;
  }
 }

 double probability_flop_at(int n, int t) const {
  if (probability_flop[n].count(t) > 0) {
    return probability_flop[n].at(t);
  } else {
    return DEFAULT_VALUE;
  }
 }

 double probability_at(int t) const {
    return probability_at(0, t) + probability_at(1, t);
}

double joint_probability(int u, int v) const {
   return probability_at(0, u) + probability_at(1, v);
}

 double last_probability() const {
   double prob_sum = 0;
   for (int n=0; n<dim; ++n) {
     prob_sum += probability[n].at(last_t[n]);
     //prob_sum += probability_at(n,last_t[n]);
   }
   return prob_sum;
    //return probability[0].at(last_t[0]) + probability[1].at(last_t[1]); //2D case
 }

 void set_probability(int i, int t, double flip_val, double flop_val) {
   probability[i][t] = logaddexp(flip_val, flop_val);
   probability_flip[i][t] = flip_val;
   probability_flop[i][t] = flop_val;
   last_t[i] = t;
 }

};

class BonitoNode : public Node<BonitoNode>{
public:
  std::unordered_map<int, double> probability;
  std::unordered_map<int, double> probability_gap;
  std::unordered_map<int, double> probability_no_gap;
  int max_t = 0;

  BonitoNode(int s, BonitoNode* p) : Node<BonitoNode>(s, p) {}
  BonitoNode(int s) : Node<BonitoNode>(s) {}
  BonitoNode() : Node<BonitoNode>() {}

   double probability_at(int t) const {
    if (probability.count(t) > 0) {
      return probability.at(t);
    } else {
      return DEFAULT_VALUE;
    }
  }

  double probability_gap_at(int t) const {
   if (probability_gap.count(t) > 0) {
     return probability_gap.at(t);
   } else {
     return DEFAULT_VALUE;
   }
 }

   double probability_no_gap_at(int t) const {
    if (probability_no_gap.count(t) > 0) {
      return probability_no_gap.at(t);
    } else {
      return DEFAULT_VALUE;
    }
  }

  double last_probability() const {
    return probability.at(max_t);
  }

  void set_probability(int t, double gap_val, double no_gap_val) {
    probability[t] = logaddexp(gap_val, no_gap_val);
    probability_gap[t] = gap_val;
    probability_no_gap[t] = no_gap_val;
    max_t = t;
  }

};

class BonitoNode2D : public Node<BonitoNode2D>{
public:
  static const int dim = 2;
  std::unordered_map<int, double> probability[dim];
  std::unordered_map<int, double> probability_gap[dim];
  std::unordered_map<int, double> probability_no_gap[dim];
  int last_t[dim] = {0, 0};

  BonitoNode2D(int s, BonitoNode2D* p) : Node<BonitoNode2D>(s, p) {}
  BonitoNode2D(int s) : Node<BonitoNode2D>(s) {}
  BonitoNode2D() : Node<BonitoNode2D>() {}

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

  double joint_probability(int u, int v) const {
     return probability_at(0, u) + probability_at(1, v);
  }

  double last_probability() const {
    double prob_sum = 0;
    for (int n=0; n<dim; ++n) {
      prob_sum += probability[n].at(last_t[n]);
      //prob_sum += probability_at(n,last_t[n]);
    }
    return prob_sum;
     //return probability[0].at(last_t[0]) + probability[1].at(last_t[1]); //2D case
  }

  void set_probability(int i, int t, double gap_val, double no_gap_val) {
    probability[i][t] = logaddexp(gap_val, no_gap_val);
    probability_gap[i][t] = gap_val;
    probability_no_gap[i][t] = no_gap_val;
    last_t[i] = t;
  }

};

template <class TNode>
class PrefixTree {
public:
    std::string alphabet;
    TNode root;

    PrefixTree(std::string a) : alphabet{a} {}
     virtual ~PrefixTree() {
        delete root;
     }

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

class PoreOverPrefixTree : public PrefixTree<PoreOverNode*> {
public:
  int gap_char;
  int t_max;
  double **y;

  PoreOverPrefixTree(double **d, int v, std::string a) : PrefixTree<PoreOverNode*>(a), y{d}, t_max{v} {
    gap_char = alphabet.length();
    root = new PoreOverNode(gap_char);
    root->probability[-1] = 0;
    double blank_sum = 0;
    for (int i=0; i<t_max; i++) {
      blank_sum += y[i][gap_char];
      root->probability[i] = blank_sum;
    }
  }

  void update_prob(PoreOverNode* n, int t) {
    double a = n->parent->probability_at(t-1);
    double b = y[t][n->last];
    double emit_state = a+b;

    double c = n->probability_at(t-1);
    double d = y[t][gap_char];
    double stay_state = c+d;

    n->set_probability(t, logaddexp(emit_state, stay_state));
  }

};

class PoreOverPrefixTree2D : public PrefixTree<PoreOverNode2D*> {
public:
  static const int dim = 2;
  int gap_char;
  int t_max[dim];
  double **y[dim];

  PoreOverPrefixTree2D(double **d1, int u, double **d2, int v, std::string a) : PrefixTree<PoreOverNode2D*>(a) {
    y[0] = d1;
    y[1] = d2;
    t_max[0] = u;
    t_max[1] = v;
    gap_char = alphabet.length();
    root = new PoreOverNode2D(gap_char);
    root->probability[0][-1] = 0;
    root->probability[1][-1] = 0;
    for (int i=0; i<dim; ++i) {
      double blank_sum = 0;
      for (int t=0; t<t_max[i]; ++t) {
        blank_sum += y[i][t][gap_char];
        root->probability[i][t] = blank_sum;
      }
    }

  }

  void update_prob(PoreOverNode2D* n, int i, int t) {

    if (n->probability[i].count(t) == 0) {
      double a = n->parent->probability_at(i, t-1);
      double b = y[i][t][n->last];
      double emit_state = a+b;

      double c = n->probability_at(i, t-1);
      double d = y[i][t][gap_char];
      double stay_state = c+d;

      n->set_probability(i, t, logaddexp(emit_state, stay_state));
  } else {
      n->set_probability(i, t);
  }
  }

};

class FlipFlopPrefixTree : public PrefixTree<FlipFlopNode*> {
public:
  int flipflop_size = alphabet.length();
  int t_max;
  double **y;

  FlipFlopPrefixTree(double **d, int v, std::string a) : PrefixTree<FlipFlopNode*>(a), y{d}, t_max{v} {
    root = new FlipFlopNode(flipflop_size);
    root->probability[-1] = 0;
    root->probability_flip[-1] = log(0.5);
    root->probability_flop[-1] = log(0.5);
  }

  void update_prob(FlipFlopNode* n, int t) {
    double stay_flip = n->probability_flip_at(t-1) + y[t][n->last];
    double stay_flop = n->probability_flop_at(t-1) + y[t][n->last + flipflop_size];

    double emit_flip, emit_flop;

    if (n->parent->depth == 0 and t==0) {
        emit_flip = y[t][n->last];
        emit_flop = y[t][n->last + flipflop_size];
    } else if (n->parent->last == n->last) {
      emit_flip = n->parent->probability_flop_at(t-1) + y[t][n->last];
      emit_flop = n->parent->probability_flip_at(t-1) + y[t][n->last + flipflop_size];
    } else {
      emit_flip = logaddexp(n->parent->probability_flip_at(t-1), n->parent->probability_flop_at(t-1)) + y[t][n->last];
      emit_flop = DEFAULT_VALUE;
    }

    double flip_prob =  logaddexp(emit_flip, stay_flip);
    double flop_prob = logaddexp(emit_flop, stay_flop);

    n->set_probability(t, flip_prob, flop_prob);

    //n->set_probability(t, logaddexp(emit_flip, stay_flip), logaddexp(emit_flop, stay_flop));
    //std::cout << "Looking at parent:" << this->get_label(n->parent) << ": probability_flip(t-1)=" << n->parent->probability_flip_at(t-1) << ": probability_flop(t-1)=" << n->parent->probability_flop_at(t-1) << "\n";
    //std::cout << y[t][n->last] << " " << y[t][n->last + flipflop_size] << "\n";
    //std::cout << this->get_label(n) << " at t=" << t << ":" <<  exp(emit_flip) << "+" << exp(stay_flip) << "," << exp(emit_flop) << "+" << exp(stay_flop) << "=" << n->probability_at(t) << "\n";
  }

};

class FlipFlopPrefixTree2D : public PrefixTree<FlipFlopNode2D*> {
public:
  static const int dim = 2;
  double **y[dim];
  int t_max[dim];
  int flipflop_size = alphabet.length();

  FlipFlopPrefixTree2D(double **d1, int u, double **d2, int v, std::string a) : PrefixTree<FlipFlopNode2D*>(a) {
    y[0] = d1;
    y[1] = d2;
    t_max[0] = u;
    t_max[1] = v;

    root = new FlipFlopNode2D(flipflop_size);
    root->probability[0][-1] = 0;
    root->probability[1][-1] = 0;
    root->probability_flip[0][-1] = log(0.5);
    root->probability_flop[0][-1] = log(0.5);
    root->probability_flip[1][-1] = log(0.5);
    root->probability_flop[1][-1] = log(0.5);
  }

  void update_prob(FlipFlopNode2D* n, int i, int t) {
    double stay_flip = n->probability_flip_at(i, t-1) + y[i][t][n->last];
    double stay_flop = n->probability_flop_at(i, t-1) + y[i][t][n->last + flipflop_size];

    double emit_flip, emit_flop;

    if (n->parent->depth == 0 and t==0) {
        emit_flip = y[i][t][n->last];
        emit_flop = y[i][t][n->last + flipflop_size];
    } else if (n->parent->last == n->last) {
      emit_flip = n->parent->probability_flop_at(i, t-1) + y[i][t][n->last];
      emit_flop = n->parent->probability_flip_at(i, t-1) + y[i][t][n->last + flipflop_size];
    } else {
      emit_flip = logaddexp(n->parent->probability_flip_at(i, t-1), n->parent->probability_flop_at(i, t-1)) + y[i][t][n->last];
      emit_flop = DEFAULT_VALUE;
    }

    double flip_prob =  logaddexp(emit_flip, stay_flip);
    double flop_prob = logaddexp(emit_flop, stay_flop);

    /*
    if (flip_prob >= flop_prob) {
        flop_prob = DEFAULT_VALUE;
    } else {
        flip_prob = DEFAULT_VALUE;
    }

    flop_prob = DEFAULT_VALUE;
    flip_prob = DEFAULT_VALUE;
    */

    n->set_probability(i, t, flip_prob, flop_prob);
  }
};

class BonitoPrefixTree : public PrefixTree<BonitoNode*> {
public:
  int t_max;
  double **y;
  int gap_char;

  BonitoPrefixTree(double **d, int v, std::string a) : PrefixTree<BonitoNode*>(a), y{d}, t_max{v} {
    gap_char = alphabet.length();
    root = new BonitoNode(gap_char);
    root->probability[-1] = 0;
    root->probability_gap[-1] = 0;
    root->probability_no_gap[-1] = DEFAULT_VALUE;
  }

  void update_prob(BonitoNode* n, int t) {
    double gap_prob = n->probability_at(t-1) + y[t][gap_char];
    double no_gap_prob;

    if (n->parent->depth == 0 and t==0) {
        no_gap_prob = y[t][n->last];
    } else if (n->parent->last == n->last) {
      no_gap_prob = logaddexp(n->parent->probability_gap_at(t-1) + y[t][n->last], n->probability_no_gap_at(t-1) + y[t][n->last]);
    } else {
      no_gap_prob = logaddexp(n->parent->probability_at(t-1) + y[t][n->last], n->probability_no_gap_at(t-1) + y[t][n->last]);
    }

    n->set_probability(t, gap_prob, no_gap_prob);

  }

};

class BonitoPrefixTree2D : public PrefixTree<BonitoNode2D*> {
public:
  static const int dim = 2;
  int t_max[dim];
  double **y[dim];
  int gap_char;

  BonitoPrefixTree2D(double **d1, int u, double **d2, int v, std::string a) : PrefixTree<BonitoNode2D*>(a) {

    y[0] = d1;
    y[1] = d2;
    t_max[0] = u;
    t_max[1] = v;
    gap_char = alphabet.length();
    root = new BonitoNode2D(gap_char);
    root->probability[0][-1] = 0;
    root->probability[1][-1] = 0;
    root->probability_gap[0][-1] = 0;
    root->probability_gap[1][-1] = 0;
    root->probability_no_gap[0][-1] = DEFAULT_VALUE;
    root->probability_no_gap[0][-1] = DEFAULT_VALUE;
  }

  void update_prob(BonitoNode2D* n, int i, int t) {
    double gap_prob = n->probability_at(i, t-1) + y[i][t][gap_char];
    double no_gap_prob;

    if (n->parent->depth == 0 and t==0) {
        no_gap_prob = y[i][t][n->last];
    } else if (n->parent->last == n->last) {
      no_gap_prob = logaddexp(n->parent->probability_gap_at(i,t-1) + y[i][t][n->last], n->probability_no_gap_at(i,t-1) + y[i][t][n->last]);
    } else {
      no_gap_prob = logaddexp(n->parent->probability_at(i,t-1) + y[i][t][n->last], n->probability_no_gap_at(i,t-1) + y[i][t][n->last]);
    }

    n->set_probability(i, t, gap_prob, no_gap_prob);

  }

};

// Use existing PrefixTree data structure to calculate forward probabilities
//  instead of explicit DP matrix. Should produce identical results.
template <class TNode, class TTree>
double forward_(double **y, int t_max, std::string label, std::string alphabet) {

    int s_max = label.length();
    int alphabet_size = alphabet.length();

    // map label string to character indices
    std::unordered_map<char, int> alphabet_map;
    for (int a=0; a<alphabet_size; a++) {
        alphabet_map[alphabet[a]] = a;
    }

    int *label_int = new int[s_max];
    for (int s=0; s<s_max; s++) {
        label_int[s] = alphabet_map[label[s]];
    }

    TTree tree(y, t_max, alphabet);
    std::vector<TNode*> substrings;

    auto currNode = tree.root;
    for (int s=0; s<s_max; s++) {
        currNode = currNode->add_child(label_int[s]);
        substrings.push_back(currNode);
        //currNode->set_probability(0, y[0][label_int[0]], y[0][label_int[0]+tree.flipflop_size]);
        for (int t=0; t<t_max; t++) {
            tree.update_prob(currNode, t);
        }
    }

    delete [] label_int;

    /*
    for (int s=0; s<s_max; s++) {
        std::cout << s << ":" << substrings[s] << " from " << substrings[s]->parent << ":" << tree.get_label(substrings[s]) << ":" << substrings[s]->last_probability() << "\n";
    }
    */

    return currNode->last_probability();
}

double forward(double **y, int t_max, std::string label, std::string alphabet, std::string model="ctc") {
    if (model == "ctc") {
        return forward_<PoreOverNode, PoreOverPrefixTree>(y, t_max, label, alphabet);
    } else if (model == "ctc_merge_repeats") {
        return forward_<BonitoNode, BonitoPrefixTree>(y, t_max, label, alphabet);
    } else if (model == "ctc_flipflop") {
        return forward_<FlipFlopNode, FlipFlopPrefixTree>(y, t_max, label, alphabet);
    }
}

#endif
