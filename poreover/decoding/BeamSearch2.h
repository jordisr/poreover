/*
C++ API:
beam_search: 1D beam

Python: beam_search_2d()
Options: model(ctc, ctc_merge_repeats, ctc_flipflop) x envelope(True, False) x method(fast, slow)

Overloading in C++ or Cython file? Model needs to live in C++, so also envelope and method

beam_search_2d() : envelope, model, method "row" or "grid"
beam_search_2d_row: single beam, faster "1.5D" search
beam_search_2d_grid: 2D beam data, prune after each (i,j) step
*/

#ifndef TEST_HPP
#define TEST_HPP

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include<set>
#include <functional>
#include <cmath>

#include "Log.h"
#include "SparseMatrix.h"
#include "PrefixTree.h"
#include "Beam.h"

template <class TTree, class TBeam>
std::string beam_search_2d_grid(double **y1, double **y2, int U, int V, std::string alphabet, int beam_width) {

  TTree tree(y1, U, y2, V, alphabet);

  TBeam* empty_beam = new TBeam(beam_width);
  //empty_beam->push(tree.root);
  auto children = tree.expand(tree.root);
  for (int i=0; i<children.size(); i++) {
      auto n = children[i];
      tree.update_prob(n, 0, 0);
      tree.update_prob(n, 1, 0);
      empty_beam->push(n);
  }

  //SparseMatrix<TBeam> beams(empty_beam);
  std::vector<TBeam*> beams;

  for (int u=0; u<U; ++u) {
    for (int v=0; v<V; ++v) {
      //std::cout << "u:" << u << "\tv:" << v << "index:"<< u*V+v<< "-------------------------------\n";

      TBeam* this_beam = new TBeam(beam_width);
      beams.push_back(this_beam);

      TBeam* prev_beam;
      if (u > 0 && v > 0) {
        prev_beam = beams[(u-1)*V+(v-1)];
        //std::cout << "Previous beam is :" << "(" << (u-1) << "," << (v-1) << ")" << "\n";
      } else {
        prev_beam = empty_beam;
        //std::cout << "Previous beam is : root\n";
      }
      //std::cout << "Beam_size:" << prev_beam->size() << "\n";
      //std::cout << "Iterating over previous beam...\n";

      for (auto beam_node : prev_beam->elements) {
            //std::cout << "label:" << tree.get_label(beam_node) << "\n";

            // update probabilities
            tree.update_prob(beam_node, 0, u);
            tree.update_prob(beam_node, 1, v);
            this_beam->push(beam_node);

            // expand node and add children to the beam
            auto children = tree.expand(beam_node);
            for (int i=0; i < children.size(); i++) {
              auto child = children[i];
              //std::cout << "Adding child:"<< tree.get_label(child) << "\n";
              tree.update_prob(child, 0, u);
              tree.update_prob(child, 1, v);
              this_beam->push(child);
            }
        }

        /*
        // write out current beam after pruning
        std::cout << "Beam before pruning:\n";
        //for (auto beam_node=this_beam->elements.rbegin(); beam_node != this_beam->elements.rend(); beam_node++) {
        for (auto beam_node : this_beam->elements) {
        std::cout << "----" << tree.get_label(beam_node) << " : "
        << (beam_node)->last_probability() << "\n";
        }
        */

        this_beam->prune();

        /*
        // write out current beam after pruning
        std::cout << "Beam after pruning:\n";
        //for (auto beam_node=this_beam->elements.rbegin(); beam_node != this_beam->elements.rend(); beam_node++) {
        for (auto beam_node : this_beam->elements) {
        std::cout << "----" << tree.get_label(beam_node) << " : "
        << (beam_node)->last_probability() << "\n";
        }

        std::cout << "-----------------\n";
        */

      }
  }

  // return top element
  auto top_node = beams[(U-1)*V+(V-1)]->top();
  return tree.get_label(top_node);

}

template <class TTree, class TBeam>
std::string beam_search_2d_grid(double **y1, double **y2, int **envelope_ranges, int U, int V, std::string alphabet, int beam_width) {

  TTree tree(y1, U, y2, V, alphabet);

  TBeam* empty_beam = new TBeam(beam_width);
  //empty_beam->push(tree.root);
  auto children = tree.expand(tree.root);
  for (int i=0; i<children.size(); i++) {
      auto n = children[i];
      tree.update_prob(n, 0, 0);
      tree.update_prob(n, 1, 0);
      empty_beam->push(n);
  }

  SparseMatrix<TBeam*> beams(empty_beam);

  for (int u=0; u<U; ++u) {

    int row_start = envelope_ranges[u][0];
    int row_end = envelope_ranges[u][1];
    beams.push_row(row_start, row_end);

    for (int v=row_start; v<row_end; ++v) {

      TBeam* this_beam = new TBeam(beam_width);
      beams.set(u, v, this_beam);

      TBeam* prev_beam;
      if (u > 0 && v > 0) {
        prev_beam = beams.get(u-1,v-1);
      } else {
        prev_beam = empty_beam;
      }

      for (auto beam_node : prev_beam->elements) {
            //std::cout << "label:" << tree.get_label(beam_node) << "\n";

            // update probabilities
            tree.update_prob(beam_node, 0, u);
            tree.update_prob(beam_node, 1, v);
            this_beam->push(beam_node);

            // expand node and add children to the beam
            auto children = tree.expand(beam_node);
            for (int i=0; i < children.size(); i++) {
              auto child = children[i];
              //std::cout << "Adding child:"<< tree.get_label(child) << "\n";
              tree.update_prob(child, 0, u);
              tree.update_prob(child, 1, v);
              this_beam->push(child);
            }
        }

        this_beam->prune();

      }
  }

  // return top element
  auto top_node = beams.get(U-1,V-1)->top();
  return tree.get_label(top_node);

}

#endif
