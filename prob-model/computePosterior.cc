#include <octave/oct.h>
#include <math.h>
#include "dec2oneOfK.cc"
#include "base2decpure.cc"

DEFUN_DLD (computePosterior, args, nargout, "") {
  // args
  NDArray mus = args(0).array_value();
  NDArray Sigmas = args(1).array_value();
  Matrix pi = args(2).array_value();
  Matrix rho = args(3).array_value();
  NDArray X = args(4).array_value();
  Matrix d = args(5).array_value();
  int K = args(6).int_value();
  int D = X.dims()(0);
  int I = X.dims()(1);
  int N = X.dims()(2);
  // compute
  long int KtoI = pow(K, I);
  Matrix p_Z(KtoI, N);
  p_Z.fill(0.0);
  for(int n = 0; n < N; n++) {
    for(int l = 0; l < KtoI; l++) {
      octave_value_list dec2_args(3);
      dec2_args(0) = l + 1;
      dec2_args(1) = K;
      dec2_args(2) = I;
      octave_value_list dec2ret = dec2oneOfKpure(dec2_args);
      Matrix Z_n = dec2ret(0).array_value();
      Matrix z = dec2ret(1).array_value();
      // select the right mus
      Matrix mus_l(D, I);
      mus_l.fill(0.0);
      Matrix z_idx(I, 1);
      for(int i = 0; i < I; i++) {
        octave_value_list base2dec_args(2);
        base2dec_args(0) = z(0, i);
        base2dec_args(1) = K;
        z_idx(i, 0) = base2decpure(base2dec_args)(0).int_value() + 1;
      }
      for(int i = 0; i < I; i++) {
        for(int d = 0; d < D; d++) {
          mus_l(d, i) = mus(d, z_idx(i), i);
        }
      }
      // select the right Sigmas
    }
  }
  // return
  return octave_value_list();
}

