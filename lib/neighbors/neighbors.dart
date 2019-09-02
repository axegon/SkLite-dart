import 'dart:math';
import 'package:sklite/base.dart';
import 'package:sklite/utils/mathutils.dart';

/// Makes life easier.
class Neighbor {
  int cls;
  double dist;

  Neighbor(this.cls, this.dist);
}

/// An implementation of sklearn.neighbors.KNeighborsClassifier
/// ---------------
///
/// https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
class KNeighborsClassifier extends Classifier {
  List<List<dynamic>> fitX;
  List<int> fitY;
  List<int> classes;
  int nNeighbors;
  int p;

  /// To manually instantiate the KNeighborsClassifier. The parameters
  /// are lifted directly from scikit-learn.
  /// See the attributes here:
  /// https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
  KNeighborsClassifier(
      this.fitX, this.fitY, this.nNeighbors, this.p, this.classes);

  factory KNeighborsClassifier.fromMap(Map<String, dynamic> params) {
    return KNeighborsClassifier(
        List<List<dynamic>>.from(params["_fit_X"]),
        List<int>.from(params["_y"]),
        params["n_neighbors"],
        params["p"],
        List<int>.from(params["classes_"]));
  }

  double cmp(List<dynamic> temp, List<dynamic> cand, int q) {
    double dist = 0.0;
    double diff = 0.0;
    for (int i = 0; i < temp.length; i++) {
      diff = (temp[i] - cand[i]).abs();
      if (q == 1)
        dist += diff;
      else if (q == 2)
        dist += diff * diff;
      else if (q == INFINITY && diff > dist)
        dist = diff;
      else
        dist += pow(diff, q);
    }
    if (q == 1 || q == INFINITY)
      return dist;
    else if (q == 2) return sqrt(dist);
    return pow(dist, 1.0 / q);
  }

  /// Implementation of sklearn.neghbors.KNeighborsClassifier.predict.
  @override
  int predict(List<double> X) {
    if (nNeighbors == 1) {
      int idx = 0;
      double minDist = INFINITY;
      double curDist;
      for (int i = 0; i < fitY.length; i++) {
        curDist = cmp(fitX[i], X, p);
        if (curDist <= minDist) {
          minDist = curDist;
          idx = fitY[i];
        }
      }
      return classes[idx];
    }
    var compute = List<int>.filled(classes.length, 0);
    var dists = [];
    fitY.asMap().forEach((i, v) => dists.add(Neighbor(v, cmp(fitX[i], X, p))));
    dists.sort((a, b) => a.dist.compareTo(b.dist));
    classes.asMap().forEach((i, v) => compute[dists[i].cls]++);

    return classes[argmax(compute)];
  }
}
