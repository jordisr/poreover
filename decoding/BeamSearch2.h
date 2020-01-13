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
//#include "SparseMatrix.h"
#include "PrefixTree.h"
#include "BeamSearch.h"
#include "Beam.h"

template <class TTree, class TBeam>
std::string new_beam_search(double **y1, double **y2, int U, int V, std::string alphabet, int beam_width) {

  TTree tree(y1, U, y2, V, alphabet);

  TBeam* empty_beam = new TBeam(beam_width);
  empty_beam->push(tree.root);

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

            // update probabilities
            //std::cout << "want to set 0 at max t=" << u << "\n";
            //std::cout << "want to set 1 at max t=" << v << "\n";
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

std::string new_beam_search_wrapper(double **y1, double **y2, int U, int V, std::string alphabet, int beam_width, std::string model="ctc") {
  return new_beam_search<PoreOverPrefixTree2D, Beam<PoreOverNode2D*>>(y1, y2, U, V, alphabet, beam_width);
}

#endif
