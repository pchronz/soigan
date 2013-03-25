#include <octave/oct.h>
#include <stdlib.h>
#include "base2decpure.cc"

DEFUN_DLD (base2decfast, args, nargout, "") {
  return base2decpure(args);
}

