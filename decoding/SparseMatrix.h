#ifndef SPARSE_H
#define SPARSE_H

#include <vector>
#include <iostream>
#include <limits>

#define DEFAULT_VALUE -std::numeric_limits<double>::infinity()

class SparseRow {
  public:
    int start, end;
    double* values;

    SparseRow(int s, int e) {
      start = s;
      end = e;
      values = new double[e-s+1];
      start = s;
      end = e;
    };

    int get_start(void) {
      return start;
    }
    int get_end(void) {
      return end;
    }

    void set(int i, double x) {
      // set single value in row
      if ((i >= start) && (i <= end)) {
        values[i-start] = x;
      } else {
        std::cout << "Not setting. Out of bounds" << std::endl;
      }
    }

    void set_values(double* x) {
      // deep copy values
      for (int i=0; i < (end-start+1); i++) {
        this->set(i+start,x[i]);
      }
    }

    const double get(int i) {
      if ((i < start) || (i > end)) {
        return DEFAULT_VALUE;
      } else {
        return values[i-start];
      }
    }
};

// class is vector of pointers to individual SparseRow objects
class SparseMatrix {
  public:
    std::vector<SparseRow*> row;
    int length;

    SparseMatrix(void) {
    length = 0;
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
      SparseRow* rowPtr = new SparseRow(start, end);
      row.push_back(rowPtr);
      for (int v=start; v<=end; v++) {
        rowPtr->set(v, DEFAULT_VALUE); // initialize row
      }
      length += 1;
    }

    void set(int i, int j, double value) {
      if ((0 <= i) && (i < length)) {
        row[i]->set(j, value);
      }
    }

    double get(int i, int j) {
      if ((0 <= i) && (i < length)) {
        return row[i]->get(j);
      } else {
        return DEFAULT_VALUE;
      }
    }
};

#endif
