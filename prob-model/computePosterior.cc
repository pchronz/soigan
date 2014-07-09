#include <octave/oct.h>
#include <iostream>
#include "math.h"
#include "dec2oneOfK.cc"
#include "base2decpure.cc"

// XXX ad-hoc implementation; should use Cholesky factorization instead
double mvnpdf(Matrix x, Matrix mu, Matrix Sigma) {
  // XXX is this Pi any good?
  long double const Pi = 4 * atan(1);
  int D = x.dims()(0);
  long double p = pow(2 * Pi, -D*0.5);
  long double det = Sigma.determinant().value();
  p *= 1.0/sqrt(det);
  long double exp = (long double)-0.5 * ((x - mu).transpose() * Sigma.inverse() * (x - mu))(0, 0);
  p *= std::exp(exp);
  // DEBUG
  if(p < 0)
    std::cerr << "The adhoc MVN implementation just returned a negative probability density" << '\n';
  return p;
}

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
  long int KtoI = std::pow(K, I);
  Matrix p_Z(KtoI, N);
  p_Z.fill(0.0);
  for(int n = 0; n < N; n++) {
    for(int l = 0; l < KtoI; l++) {
      octave_value_list dec2_args(3);
      dec2_args(0) = l + 1;
      dec2_args(1) = K;
      dec2_args(2) = I;
      octave_value_list dec2ret = dec2oneOfKpure(dec2_args);
      boolMatrix Z_n = dec2ret(0).bool_array_value();
      charMatrix z = dec2ret(1).char_array_value();
      // select the right mus
      Matrix mus_l(D, I);
      mus_l.fill(0.0);
      Matrix z_idx(I, 1);
      for(int i = 0; i < I; i++) {
        octave_value_list base2dec_args(2);
        base2dec_args(0) = z(0, i);
        base2dec_args(1) = K;
        z_idx(i, 0) = base2decpure(base2dec_args)(0).int_value();
      }
      for(int i = 0; i < I; i++) {
        for(int d = 0; d < D; d++) {
          mus_l(d, i) = mus(d, z_idx(i), i);
        }
      }
      // select the right Sigmas
      dim_vector Sigmas_l_dim = dim_vector::alloc(3);
      Sigmas_l_dim(0) = D;
      Sigmas_l_dim(1) = D;
      Sigmas_l_dim(2) = I;
      NDArray Sigmas_l(Sigmas_l_dim);
      Sigmas_l.fill(0.0);
      for(int i = 0; i < I; i++) {
        for(int d1 = 0; d1 < D; d1++) {
          for(int d2 = 0; d2 < D; d2++) {
            Matrix Sigmas_idx(1, 4);
            Sigmas_idx(0, 0) = d1;
            Sigmas_idx(0, 1) = d2;
            Sigmas_idx(0, 2) = z_idx(i, 0);
            Sigmas_idx(0, 3) = i;
            Sigmas_l(d1, d2, i) = Sigmas(Sigmas_idx);
          }
        }
      }
      // select the right pis
      Matrix pi_l(1, I);
      pi_l.fill(0.0);
      for(int i = 0; i < I; i++) {
        pi_l(0, i) = pi(z_idx(i, 0), i);
      }
      // compute the posterior for the current state and observation
      p_Z(l, n) = log(rho(l, 0))*d(0, n) + log(1 - rho(l, 0))*(1 - d(0, n));
      for(int i = 0; i < I; i++) {
        Matrix x_n_i(D, 1);
        for(int d = 0; d < D; d++) {
          Matrix x_idx = Matrix(1, 3);
          x_idx(0, 0) = d;
          x_idx(0, 1) = i;
          x_idx(0, 2) = n;
          x_n_i(d, 0) = X(x_idx);
        }
        Matrix mu_l_i = mus_l.column(i);
        Matrix Sigma_l_i(D, D);
        for(int d1 = 0; d1 < D; d1++) {
          for(int d2 = 0; d2 < D; d2++) {
            Matrix Sigmas_idx(1, 3);
            Sigmas_idx(0, 0) = d1;
            Sigmas_idx(0, 1) = d2;
            Sigmas_idx(0, 2) = i;
            Sigma_l_i(d1, d2) = Sigmas_l(Sigmas_idx);
          }
        }
        long double p_x_n_i = mvnpdf(x_n_i, mu_l_i, Sigma_l_i);
        p_Z(l, n) += log(pi_l(0, i)) + log(p_x_n_i);
      }
      // Un-log
      p_Z(l, n) = exp(p_Z(l, n));
    }
  }
  //p_Z = p_Z * p_Z.sum().diag().inverse();
  // return
  octave_value_list retval(1);
  retval(0) = p_Z;
  return retval;
}

