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

    SparseRow(int s, int e) {
      start = s;
      end = e;
      values = new T[e-s+1];
      start = s;
      end = e;
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
        return -std::numeric_limits<T>::infinity();
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

    SparseMatrix() :length{0}, default_value{-std::numeric_limits<T>::infinity()} {}
    SparseMatrix(T d) :length{0}, default_value{d} {}

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

    void push_row(int start, int end) {
      //std::cout << "Pushing row from " << start << " to " << end << std::endl;
      SparseRow<T>* rowPtr = new SparseRow<T>(start, end);
      row.push_back(rowPtr);
      for (int v=start; v<=end; v++) {
        rowPtr->set(v, default_value);
      }
      length += 1;
    }

    void set(int i, int j, T value) {
      if ((0 <= i) && (i < length)) {
        row[i]->set(j, value);
      }
    }

    T get(int i, int j) {
      if ((0 <= i) && (i < length)) {
        return row[i]->get(j);
      } else {
        return default_value;
      }
    }
};

#endif
