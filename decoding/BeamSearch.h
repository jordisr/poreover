#ifndef BEAM_SEARCH_HPP
#define BEAM_SEARCH_HPP

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <functional>
#include <cmath>

//class Node;
//class FlipFlopNode;
//class PoreOverPrefixTree;
//class FlipFlopPrefixTree;

#include "Log.h"
#include "PrefixTree.h"

template <class T>
class Beam {
  public:
    int width;
    std::vector<T> elements;

    Beam(int w): width{w} {}

    void push(T n) {
      elements.push_back(n);
    }

    void push(std::vector<T> n_vector) {
      for (int i=0; i < n_vector.size(); i++) {
        elements.push_back(n_vector[i]);
      }
    }

    int size() {
      return elements.size();
    }

    void prune() {
        // sort elements, eliminate duplicates, then prune beam to top W
        std::sort(elements.begin(), elements.end()); // sorting twice to remove duplicate elements
        auto last = std::unique(elements.begin(), elements.end());
        elements.erase(last, elements.end());
        //std::partial_sort(elements.begin(), elements.begin()+width, elements.end(), node_greater);
        std::sort(elements.begin(), elements.end(), node_greater<T>);
        if (elements.size() > width) {
          int element_size = elements.size();
          for (int i=width; i < element_size; i++) {
            elements.pop_back();
          }
      }
    }

    T top() {
      return elements[0];
    }

};

template <class TTree, class TBeam>
std::string beam_search_(double **y, int t_max, std::string alphabet, int beam_width) {

  TTree tree(y, t_max, alphabet);
  TBeam beam_(beam_width);

  // first iteration
  auto children = tree.expand(tree.root);
  for (int i=0; i<children.size(); i++) {
    auto n = children[i];
    tree.update_prob(n, 0);
    beam_.push(n);
  }

  // iterate over each node in beam
  for (int t=1; t < t_max; t++) {
    int beam_size = beam_.size();
    for (int b=0; b < beam_size; b++) {
      auto beam_node = beam_.elements[b];

      // update probabilities
      tree.update_prob(beam_node, t);

      // expand node and add children to the beam
      auto children = tree.expand(beam_node);
      for (int i=0; i<children.size(); i++) {
        auto child = children[i];
        tree.update_prob(child, t);
        beam_.push(child);
      }
    }

    // take top beam_width nodes
    beam_.prune();

  }

  // return top element
  auto top_node = beam_.top();
  return tree.get_label(top_node);
}

std::string beam_search(double **y, int t_max, std::string alphabet, int beam_width, bool flipflop=false) {
  if (flipflop) {
    return beam_search_<FlipFlopPrefixTree, Beam<FlipFlopNode*>>(y, t_max, alphabet, beam_width);
  } else {
    return beam_search_<PoreOverPrefixTree, Beam<PoreOverNode*>>(y, t_max, alphabet, beam_width);
  }
}

std::string beam_search_2d(double **y1, double **y2, int **envelope_ranges, int U, int V, std::string alphabet) {

  const int beam_width = 25;

  PoreOverPrefixTree2D tree(y1, U, y2, V, alphabet);
  Beam<PoreOverNode2D*> beam_(beam_width);

  // first iteration, check bounds for (0,0)?
  auto children = tree.expand(tree.root);
  for (int i=0; i<children.size(); i++) {
    auto n = children[i];
    tree.update_prob(n, 0, 0);
    tree.update_prob(n, 1, 0);
    beam_.push(n);
  }

  for (int u=0; u<U; ++u) {
      std::cout << "Starting row " << u << "/" << U << "\n";
    int row_start = envelope_ranges[u][0];
    int row_end = envelope_ranges[u][1];
    for (int v=row_start; v<row_end; ++v) {

      for (int b=0; b < beam_width; ++b) {
        auto beam_node = beam_.elements[b];

        // update probabilities
        tree.update_prob(beam_node, 0, u);
        tree.update_prob(beam_node, 1, v);

        // expand node and add children to the beam
        auto children = tree.expand(beam_node);
        for (int i=0; i<children.size(); i++) {
          auto child = children[i];
          tree.update_prob(child, 0, u);
          tree.update_prob(child, 1, v);
          beam_.push(child);
        }
      }

      // take top beam_width nodes
      beam_.prune();

      }
    }

  /*
  // write out beam
  for (int b=0; b < beam_.size(); b++) {
    std::cout << "----" << tree.get_label(beam_.elements[b]) << " : "
              << beam_.elements[b]->joint_probability(U-1,V-1) << "\n";
  }
  */

  auto top_node = beam_.top();
  return tree.get_label(top_node);

}



#endif