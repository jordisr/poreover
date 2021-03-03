#ifndef BEAM_HPP
#define BEAM_HPP

#include <set>
#include <vector>
#include <unordered_set>
#include <iterator>
#include <cmath>

template <class T>
class node_greater_beam {
// store beam score explicitly in node instead of calculating during comparison
public:
  bool operator()(T x, T y) {
    auto lhs = x->beam_score;
    auto rhs = y->beam_score;
    return (lhs > rhs);
  }
};

template <class T>
class node_greater {
public:
  bool operator()(T x, T y) {
    auto lhs = x->last_probability();
    auto rhs = y->last_probability();
    return (lhs > rhs);
  }
};

template <class T>
class node_greater_max {
public:
  bool operator()(T x, T y) {
    auto lhs = x->max_probability();
    auto rhs = y->max_probability();
    return (lhs > rhs);
  }
};

template <class T>
class node_greater_max_sym {
public:
  bool operator()(T x, T y) {
    auto lhs = x->max_probability_sym();
    auto rhs = y->max_probability_sym();
    return (lhs > rhs);
  }
};

template <class T>
class node_greater_max_lengthnorm {
// length normalized
public:
  bool operator()(T x, T y) {
    //float length_norm_x = std::pow(x->depth, 0.05);
    float length_norm_x = std::pow((50 + x->depth), 0.6)/std::pow((50 + 1), 0.6);
    float length_norm_y = std::pow((50 + y->depth), 0.6)/std::pow((50 + 1), 0.6);
    //float length_norm_x = std::pow(std::abs(1000-x->depth),0.05)+1;
    //float length_norm_x = std::pow(std::abs(x->depth),0.05)+1;
    //float length_norm_y = std::pow(y->depth, 0.05);
    //float length_norm_y = std::pow(std::abs(1000-y->depth),0.05)+1;
    //float length_norm_y = std::pow(std::abs(y->depth),0.05)+1;
    auto lhs = x->max_probability()/length_norm_x;
    auto rhs = y->max_probability()/length_norm_y;
    return (lhs > rhs);
  }
};

template <class T>
bool node_greater_fun(T x, T y) {
    auto lhs = x->last_probability();
    auto rhs = y->last_probability();
    return (lhs > rhs);
}

template <class T>
bool node_greater_fun_max(T x, T y) {
    auto lhs = x->max_probability();
    auto rhs = y->max_probability();
    return (lhs > rhs);
}

// T is node type (e.g. PoreOverNode2D), F is comparator functor for sorting beam (e.g. node_greater)
template <class T, class F>
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
        // first sort pointers to eliminate duplicates with std::unique
        std::sort(elements.begin(), elements.end());
        auto last = std::unique(elements.begin(), elements.end());
        //std::cout << "Removing " << elements.end() - last << " duplicates out of " << elements.size() << "\n";
        elements.erase(last, elements.end());

        // next sort nodes using comparator and take top beam width ones
        if (elements.size() > width) {
          std::partial_sort(elements.begin(), elements.begin()+width, elements.end(), F());
          elements.erase(elements.begin()+width, elements.end());
        } else {
          std::sort(elements.begin(), elements.end(), F());
        }
    }

    T top() {
      return elements[0];
    }

};

template <class T, class F>
class Beam2 {
    /*
    Beam implementation using std::set as backend
    */
  public:
    int width;
    std::set<T, F> elements;
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
