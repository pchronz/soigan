#include <octave/oct.h>
#include <octave/parse.h>
#include <math.h>
#include "dec2oneOfKpure.cc"

DEFUN_DLD (maxSigmas, args, nargout, "")
{
  // args
  NDArray X = args(0).array_value();
  NDArray mus = args(1).array_value();
  NDArray p_Z = args(2).array_value();
  // compute
  int D = X.dims()(0);
  int I = X.dims()(1);
  int N = X.dims()(2);
  int K = mus.dims()(1);
  dim_vector Sigmas_dims = dim_vector::alloc(4);
  Sigmas_dims(0) = D;
  Sigmas_dims(1) = D;
  Sigmas_dims(2) = K;
  Sigmas_dims(3) = I;
  NDArray Sigmas(Sigmas_dims);
  for(int i = 0; i < I; i++) {
    for(int k = 0; k < K; k++) {
      double Sigma_norm = 0;
      for(int n = 0; n < N; n++) {
        dim_vector diff_dim = dim_vector::alloc(3);
        diff_dim(0) = D;
        diff_dim(1) = I;
        diff_dim(2) = N;
        NDArray diff_n(diff_dim);
        for(int d = 0; d < D; d++) {
          for(int ix = 0; ix < D; ix++) {
            Matrix X_idx(1, 3);
            X_idx(0) = d;
            X_idx(1) = ix;
            X_idx(2) = n;
            Matrix mus_idx(1, 3);
            mus_idx(0, 0) = d;
            mus_idx(0, 1) = k;
            mus_idx(0, 2) = ix;
            diff_n(X_idx) = X(X_idx) - mus(mus_idx);
          }
        }
        long int KtoI = pow(K, I);
        for(long int l = 0; l < KtoI; l++) {
          octave_value_list dec2args(3);
          dec2args(0) = l + 1;
          dec2args(1) = K;
          dec2args(2) = I;
          octave_value_list dec2res = dec2oneOfKpure(dec2args);
          boolMatrix Z_n = dec2res(0).bool_array_value();
          charMatrix z = dec2res(1).char_array_value();
          for(int d1 = 0; d1 < D; d1++) {
            for(int d2 = 0; d2 < D; d2++) {
              Matrix Sigmas_idx(1, 4);
              Sigmas_idx(0) = d1;
              Sigmas_idx(1) = d2;
              Sigmas_idx(2) = k;
              Sigmas_idx(3) = i;
              double z_n_k_i = Z_n(k, i) ? 1.0 : 0.0;
              Matrix diff_idx(1, 3);
              diff_idx(0) = d1;
              diff_idx(1) = d2;
              diff_idx(2) = n;
              Sigmas(Sigmas_idx) += p_Z(l, n) * z_n_k_i * diff_n(diff_idx);
              Sigma_norm += p_Z(l, n) * z_n_k_i;
            }
          }
        }
      }
    }
  }
  // return
  octave_value_list retval(1);
  retval(0) = Sigmas;
  return retval;
}

