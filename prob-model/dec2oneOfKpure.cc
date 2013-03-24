#include <octave/oct.h>
#include <octave/parse.h>

// transform the decimal number into the matrix Z_n with column vectors coded as 1-of-K
DEFUN_DLD(dec2oneOfK, args, nargout, "") 
{
  // first create a matrix of column vectors representing all states
  int l = args(0).array_value()(0);
  int K = args(1).array_value()(0);
  int I = args(2).array_value()(0);
  Matrix k(K, I);
  for(int r = 0; r < K; r++) {
    for(int c = 0; c < I; c++) {
      k(r, c) = r;
    }
  }
  octave_value_list dec2base_args(2);
  dec2base_args(0) = k;
  dec2base_args(1) = K;
  charMatrix k_char = feval(std::string("dec2base"), dec2base_args, 1)(0).char_array_value();
  k_char = k_char.reshape(K, I);
  // then create a matrix representing the state of each vector K times
  dec2base_args = octave_value_list(3);
  dec2base_args(0) = l-1;
  dec2base_args(1) = K;
  dec2base_args(2) = I;
  charMatrix z_1 = feval(std::string("dec2base"), dec2base_args, 1)(0).char_array_value();
  charMatrix z(K, I);
  boolMatrix Z(K, I);
  for(int r = 0; r < K; r++) {
    for(int c = 0; c < I; c++) {
      z(r, c) = z_1(0, c);
      // now compare both matrices to find out where the right states are selected i.e. which of all the states (k) is selected (z)
      Z(r, c) = k_char(r, c) == z_1(c);
    }
  }
  octave_value_list retval(2);
  retval(0) = Z;
  retval(1) = z;
  return retval;
}


