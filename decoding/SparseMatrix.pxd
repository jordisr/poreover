cdef extern from "SparseMatrix.h":
    pass

# Declare the class with cdef
cdef extern from "SparseMatrix.h":
    cdef cppclass SparseMatrix:
        int length;
        void push_row(int, int);
        void set(int,int,double);
        double get(int,int);
