#include <octave/oct.h>

/**
 * C++ version 0.4 std::string style "itoa":
 * Contributions from Stuart Lowe, Ray-Yuan Sheu,
 * Rodrigo de Salvo Braz, Luc Gallant, John Maloney
 * and Brian Hunt
 */
std::string itoa(int value, int base) {

    std::string buf;

    // check that the base if valid
    if (base < 2 || base > 16) return buf;

    enum { kMaxDigits = 35 };
    buf.reserve( kMaxDigits ); // Pre-allocate enough space.

    int quotient = value;

    // Translating number to string with base:
    do {
        buf += "0123456789abcdef"[ std::abs( quotient % base ) ];
        quotient /= base;
    } while ( quotient );

    // Append the negative sign
    if ( value < 0) buf += '-';

    std::reverse( buf.begin(), buf.end() );
    return buf;
}

// transform the decimal number into the matrix Z_n with column vectors coded as 1-of-K
octave_value_list dec2oneOfKpure(octave_value_list args) {
  // first create a matrix of column vectors representing all states
  int l = args(0).array_value()(0);
  int K = args(1).array_value()(0);
  int I = args(2).array_value()(0);
  charMatrix k(K, I);
  for(int r = 0; r < K; r++) {
    for(int c = 0; c < I; c++) {
      std::string r_str = itoa(r, K);
      k(r, c) = r_str[0];
    }
  }
  octave_value_list dec2base_args(2);
  // then create a matrix representing the state of each vector K times
  charMatrix z_1(1, I);
  z_1.fill('0');
  std::string z_c = itoa(l-1, K);
  for(int i = I - 1; i >= 0; i--) {
    int diff = I - z_c.size();
    if(i - diff >= 0)
      z_1(0, i) = z_c[i - diff];
    else 
      break;
  }
  charMatrix z(K, I);
  boolMatrix Z(K, I);
  for(int r = 0; r < K; r++) {
    for(int c = 0; c < I; c++) {
      z(r, c) = z_1(c, 0);
      // now compare both matrices to find out where the right states are selected i.e. which of all the states (k) is selected (z)
      Z(r, c) = k(r, c) == z_1(c);
    }
  }
  octave_value_list retval(2);
  retval(0) = Z;
  retval(1) = z;
  return retval;
}


