#include <octave/oct.h>
#include <stdlib.h>

octave_value_list base2decpure(octave_value_list args) {
  charMatrix z = args(0).char_array_value();
  char* z_char = new char[z.length() + 1];
  for(int i = 0; i < z.length(); i++) {
    z_char[i] = z(i, 0);
  }
  z_char[z.length()] = '\0';
  int base = args(1).int_value();
  int l = strtol(z_char, NULL, base);
  octave_value_list retval(1);
  retval(0) = l;
  return retval;
}

