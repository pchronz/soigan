#include <octave/oct.h>
#include "dec2oneOfKpure.cc"

DEFUN_DLD (dec2oneOfK, args, nargout, "") {
  return dec2oneOfKpure(args);
}


