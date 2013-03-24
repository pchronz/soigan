#include <octave/oct.h>
#include <octave/parse.h>
#include <math.h>
#include "dec2oneOfK.cc"

DEFUN_DLD (maxMuPi, args, nargout, "")
{
  // args
  NDArray p_Z = args(0).array_value();
  NDArray X = args(1).array_value();
  int K = args(2).array_value()(0);
  int D = X.dims()(0);
  int I = X.dims()(1);
  int N = X.dims()(2);
  // init
  Matrix pi(K, I);
  pi.fill(0.0);
  dim_vector DKI = dim_vector::alloc(3);
  DKI(0) = D;
  DKI(1) = K;
  DKI(2) = I;
  NDArray mus(DKI);
  mus.fill(0.0);
  // maximize
  int KtoI = std::pow(K, I);
  for(int i = 0; i < I; i++) {
    for(int k = 0; k < K; k++) {
      // mus
      double mu_norm = 0.0;
      for(int n = 0; n < N; n++) {
        for(int l = 0; l < KtoI; l++) {
          octave_value_list dec_conv_args(3);
          dec_conv_args(0) = l + 1;
          dec_conv_args(1) = K;
          dec_conv_args(2) = I;
          // octave_value_list dec2_res = feval("dec2oneofK", dec_conv_args, 2);
          octave_value_list dec2_res = dec2oneOfK(dec_conv_args);
          boolMatrix Z_n = dec2_res(0).bool_array_value();
          charMatrix z = dec2_res(0).char_array_value();
          // pi
          pi(k, i) += Z_n(k, i) * p_Z(l, n);
          // mus
          for(int d = 0; d < D; d++) {
            mus(d, k, i) += p_Z(l, n) * Z_n(k, i) * X(d, i, n);
          }
          mu_norm += p_Z(l, n) * Z_n(k, i);
        }
      }
      // norm
      pi(k, i) /= N;
      for(int d = 0; d < D; d++) {
        mus(d, k, i) /= mu_norm;
      }
    }
  }
  // return
  octave_value_list retval(2);
  retval(0) = mus;
  retval(1) = pi;
  return retval;
}

