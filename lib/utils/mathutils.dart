import 'dart:math';

const double INFINITY = 1.0 / 0.0;
const double NEGATIVE_INFINITY = INFINITY * -1;

int argmax(List<dynamic> X) {
  int idx = 0;
  int l = X.length;
  for (int i = 0; i < l; i++) {
    idx = X[i] > X[idx] ? i : idx;
  }
  return idx;
}

/// Oddly enough dart does not have hyperbolic
/// functions built-in (yet?).
double tanh(double X) {
  return -1.0 + 2.0 / (1 + pow(e, (-2 * X)));
}
