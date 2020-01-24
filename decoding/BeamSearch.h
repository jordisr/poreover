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

#include "Log.h"
#include "Beam.h"
#include "PrefixTree.h"

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

// update beam after each move, still testing
template <class TTree, class TBeam>
std::string beam_search_2d_by_node(double **y1, double **y2, int **envelope_ranges, int U, int V, std::string alphabet, int beam_width) {

  TTree tree(y1, U, y2, V, alphabet);
  TBeam beam_(beam_width);

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

  auto top_node = beam_.top();
  return tree.get_label(top_node);

}

template <class TTree, class TBeam>
std::string beam_search_2d_by_row(double **y1, double **y2, int **envelope_ranges, int U, int V, std::string alphabet, int beam_width) {

    TTree tree(y1, U, y2, V, alphabet);
    TBeam beam_(beam_width);

    // first iteration, check bounds for (0,0)?
    auto children = tree.expand(tree.root);
    for (int i=0; i<children.size(); i++) {
        auto n = children[i];
        tree.update_prob(n, 0, 0);
        tree.update_prob(n, 1, 0);
        beam_.push(n);
    }

    std::cout << "u" << "\t" << "v_start" << "\t" << "v_end" << "\t" << "top_node" << "\t" << "max_probability" << "\t" << "max_t" << "\t" << "length"<< "\n";

    for (int u=0; u<U; ++u) {
        // std::cout << "Starting row " << u << "/" << U << "\n";
        int row_start = envelope_ranges[u][0];
        int row_end = envelope_ranges[u][1];

        for (int b=0; b < beam_width; ++b) {

            auto beam_node = beam_.elements[b];
            tree.update_prob(beam_node, 0, u);

            // expand node and add children to the beam
            auto children = tree.expand(beam_node);
            for (int i=0; i<children.size(); i++) {
                auto child = children[i];
                tree.update_prob(child, 0, u);
                beam_.push(child);
            }
        }

        for (int v=row_start; v<row_end; ++v) {

            for (int b=0; b < beam_.size(); b++) {
                auto beam_node = beam_.elements[b];
                if (v == row_start) {
                  beam_node->reset_max();
                }
                tree.update_prob(beam_node, 1, v);
            }

        }

        // take top beam_width nodes
        beam_.prune();

        // just output statistics from top node
        auto top_node_ = beam_.top();
        std::string top_node_label = tree.get_label(top_node_);
        //std::cout << "u=" << u << ", v=(" << row_start << "," << row_end << ")" << "----" << top_node_ << " : " << top_node_->max_probability() << "\tmax[1] at v=" << top_node_->max_t[1] << "\n";
        std::cout << u << "\t" << row_start << "\t" << row_end << "\t" << top_node_ << "\t" << top_node_->max_probability() << "\t" << top_node_->max_t[1] << "\t" << top_node_->depth << "\n";

}

auto top_node = beam_.top();
return tree.get_label(top_node);

}

// overloaded without alignment envelope
template <class TTree, class TBeam>
std::string beam_search_2d_by_row(double **y1, double **y2, int U, int V, std::string alphabet, int beam_width) {

    TTree tree(y1, U, y2, V, alphabet);
    TBeam beam_(beam_width);

    // first iteration, check bounds for (0,0)?
    auto children = tree.expand(tree.root);
    for (int i=0; i<children.size(); i++) {
        auto n = children[i];
        tree.update_prob(n, 0, 0);
        tree.update_prob(n, 1, 0);
        beam_.push(n);
    }

    std::cout << "u" << "\t" << "v_start" << "\t" << "v_end" << "\t" << "top_node" << "\t" << "max_probability" << "\t" << "max_t" << "\t" << "length"<< "\n";

    for (int u=0; u<U; ++u) {
        //std::cout << "Starting row " << u << "/" << U << "\n";
        for (int b=0; b < beam_width; ++b) {
            auto beam_node = beam_.elements[b];
            tree.update_prob(beam_node, 0, u);

            // expand node and add children to the beam
            auto children = tree.expand(beam_node);
            for (int i=0; i<children.size(); i++) {
                auto child = children[i];
                tree.update_prob(child, 0, u);
                beam_.push(child);
            }
        }

        for (int v=0; v<std::min(u,V); ++v) {
            for (int b=0; b < beam_.size(); b++) {
                auto beam_node = beam_.elements[b];
                if (v == 0) {
                  beam_node->reset_max();
                }
                tree.update_prob(beam_node, 1, v);
            }

        }

        // take top beam_width nodes
        beam_.prune();

        // write out beam
        //for (int b=0; b < beam_.size(); b++) {
        //std::cout << "u=" << u << ", v=(0," << V << ")" << "----" << tree.get_label(beam_.elements[b]) << " : " << beam_.elements[b]->max_probability() << "\tmax[1] at v=" << beam_.elements[b]->max_t[1] << "\n";
        //}

        // just output statistics from top node
        auto top_node_ = beam_.top();
        std::string top_node_label = tree.get_label(top_node_);
        std::cout << u << "\t" << 0 << "\t" << V << "\t" << top_node_ << "\t" << top_node_->max_probability() << "\t" << top_node_->max_t[1] << "\t" << top_node_->depth << "\n";
        //std::cout << "u=" << u << ", v=(0," << V << ")" << "----" << top_node_ << " : " << top_node_->max_probability() << "\tmax[1] at v=" << top_node_->max_t[1] << "\n";

}

/*
// write out final beam
for (int b=0; b < beam_.size(); b++) {
std::cout << "----" << tree.get_label(beam_.elements[b]) << " : "
<< beam_.elements[b]->joint_probability(U-1,V-1) << "\n";
}
*/

auto top_node = beam_.top();
return tree.get_label(top_node);

}

// beam search on single read
std::string beam_search(double **y, int t_max, std::string alphabet, int beam_width, std::string model="ctc") {
    if (model == "ctc") {
        return beam_search_<PoreOverPrefixTree, Beam<PoreOverNode*, node_greater<PoreOverNode*>>>(y, t_max, alphabet, beam_width);
    } else if (model == "ctc_merge_repeats") {
        return beam_search_<BonitoPrefixTree, Beam<BonitoNode*, node_greater<BonitoNode*>>>(y, t_max, alphabet, beam_width);
    } else if (model == "ctc_flipflop") {
        return beam_search_<FlipFlopPrefixTree, Beam<FlipFlopNode*, node_greater<FlipFlopNode*>>>(y, t_max, alphabet, beam_width);
    }
}

// pair beam search with envelope
std::string beam_search(double **y1, double **y2, int U, int V, std::string alphabet, int **envelope_ranges, int beam_width, std::string model="ctc") {
    if (model == "ctc") {
        return beam_search_2d_by_row<PoreOverPrefixTree2D, Beam<PoreOverNode2D*, node_greater_max<PoreOverNode2D*>>>(y1, y2, envelope_ranges, U, V, alphabet, beam_width);
    } else if (model == "ctc_merge_repeats") {
        return beam_search_2d_by_row<BonitoPrefixTree2D, Beam<BonitoNode2D*, node_greater_max<BonitoNode2D*>>>(y1, y2, envelope_ranges, U, V, alphabet, beam_width);
    } else if (model == "ctc_flipflop") {
        return beam_search_2d_by_row<FlipFlopPrefixTree2D, Beam<FlipFlopNode2D*, node_greater_max<FlipFlopNode2D*>>>(y1, y2, envelope_ranges, U, V, alphabet, beam_width);
    }
}

// pair beam search without envelope
std::string beam_search(double **y1, double **y2, int U, int V, std::string alphabet, int beam_width, std::string model="ctc") {
    if (model == "ctc") {
        return beam_search_2d_by_row<PoreOverPrefixTree2D, Beam<PoreOverNode2D*, node_greater_max<PoreOverNode2D*>>>(y1, y2, U, V, alphabet, beam_width);
    } else if (model == "ctc_merge_repeats") {
        return beam_search_2d_by_row<BonitoPrefixTree2D, Beam<BonitoNode2D*, node_greater_max<BonitoNode2D*>>>(y1, y2, U, V, alphabet, beam_width);
    } else if (model == "ctc_flipflop") {
        return beam_search_2d_by_row<FlipFlopPrefixTree2D, Beam<FlipFlopNode2D*, node_greater_max<FlipFlopNode2D*>>>(y1, y2, U, V, alphabet, beam_width);
    }
}

#endif
