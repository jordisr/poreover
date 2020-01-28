#ifndef SPARSE_H
#define SPARSE_H

#include <vector>
#include <iostream>
#include <limits>

template <class T>
class SparseRow {
  public:
    int start, end;
    T* values;
    T default_value;

    SparseRow(int s, int e, T d) {
      start = s;
      end = e;
      values = new T[e-s+1];
      start = s;
      end = e;
      default_value = d;
    };

    ~SparseRow() {
        delete [] values;
    }

    int get_start(void) {
      return start;
    }
    int get_end(void) {
      return end;
    }

    void set(int i, T x) {
      // set single value in row
      if ((i >= start) && (i <= end)) {
        values[i-start] = x;
      } else {
        //std::cout << "Not setting. Out of bounds" << std::endl;
      }
    }

    void set_values(T* x) {
      // deep copy values
      for (int i=0; i < (end-start+1); i++) {
        this->set(i+start,x[i]);
      }
    }

    const T get(int i) {
      if ((i < start) || (i > end)) {
        return default_value;
      } else {
        return values[i-start];
      }
    }
};

// class is vector of pointers to individual SparseRow objects
template <class T>
class SparseMatrix {
  public:
    std::vector<SparseRow<T>*> row;
    int length;
    T default_value;

    SparseMatrix(T d) :length{0}, default_value{d} {}
    // if no default value provided, try to use -infinity
    SparseMatrix() :length{0}, default_value{-std::numeric_limits<T>::infinity()} {}

    ~SparseMatrix() {
        for (auto x : row) {
            delete x;
        }
    }

    bool contains(int i, int j) {
      if ((0 <= i) && (i < length)){
        if ((row[i]->start <= j) && (j <= row[i]->end)) {
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
    }

    void push_row(int start, int end, T value) {
      // set row to some value
      SparseRow<T>* rowPtr = new SparseRow<T>(start, end, default_value);
      row.push_back(rowPtr);
      for (int v=start; v<=end; v++) {
        rowPtr->set(v, value);
      }
      length += 1;
    }

    void push_row(int start, int end) {
      push_row(start, end, default_value);
    }

    void set(int i, int j, T value) {
      if ((0 <= i) && (i < length)) {
        row[i]->set(j, value);
      }
    }

    const T get(int i, int j) {
      if ((0 <= i) && (i < length)) {
        return row[i]->get(j);
      } else {
        return default_value;
      }
    }
};

#endif
