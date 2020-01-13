#ifndef BEAM_HPP
#define BEAM_HPP

#include <set>
#include <vector>
#include <unordered_set>
#include <iterator>

template <class T>
class node_greater {
public:
  bool operator()(T x, T y) {
    auto lhs = x->last_probability();
    auto rhs = y->last_probability();
    //std::cout << lhs << " < " << rhs << " = " << (lhs < rhs) << "\n";
    return (lhs > rhs);
  }
};

template <class T>
bool node_greater_fun(const T x, const T y) {
    auto lhs = x->last_probability();
    auto rhs = y->last_probability();
    return (lhs > rhs);
}

template <class T>
class Beam {
    /*
    Previous Beam implementation using std::vector and explicit sort calls
    */
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
        //std::cout << "Removing " << elements.end() - last << " duplicates out of " << elements.size() << "\n";
        elements.erase(last, elements.end());
        /*
        std::unordered_set<T> s;
        for (auto i : elements)
            s.insert(i);
        elements.assign( s.begin(), s.end() );
        */

        //std::sort(elements.begin(), elements.end(), node_greater<T>);
        if (elements.size() > width) {
          std::partial_sort(elements.begin(), elements.begin()+width, elements.end(), node_greater_fun<T>);
          elements.erase(elements.begin()+width, elements.end());
          /*
          int element_size = elements.size();
          for (int i=width; i < element_size; i++) {
            elements.pop_back();
          }
          */
      }
    }

    T top() {
      return elements[0];
    }

};

template <class T>
class Beam2 {
    /*
    Beam implementation using std::set as backend
    */
  public:
    int width;
    std::set<T, node_greater<T>> elements;
    //std::set<T> elements;

    Beam2(int w): width{w} {}

    void push(T n) {
      elements.insert(n);
    }

    void push(std::vector<T> n_vector) {
      for (int i=0; i < n_vector.size(); i++) {
        elements.insert(n_vector[i]);
      }
    }

    int size() {
      return elements.size();
    }

    void prune() {
        auto it = elements.begin();
        int num_save = 0;
        while (it != elements.end() && num_save < width) {
            //std::cout << "Keeping... " << (*it)->last_probability() << "\n";
            it++;
            num_save++;
        }
        elements.erase(it, elements.end());
    }

    T top() {
      return *elements.begin();
    }

};

template <class T>
class Beam3 {
    /*
    CONTROL TEST FOR BENCHMARKING -- UNSORTED VECTOR
    */
  public:
    int width;
    std::vector<T> elements;

    Beam3(int w): width{w} {}

    void push(T n) {
      elements.push_back(n);
    }

    void push(std::vector<T> n_vector) {
      for (int i=0; i < n_vector.size(); i++) {
        elements.insert(n_vector[i]);
      }
    }

    int size() {
      return elements.size();
    }

    void prune() {
        auto it = elements.begin();
        int num_save = 0;
        while (it != elements.end() && num_save < width) {
            //std::cout << "Keeping... " << (*it)->last_probability() << "\n";
            it++;
            num_save++;
        }
        elements.erase(it, elements.end());
    }

    T top() {
      return *elements.begin();
    }

};

#endif
