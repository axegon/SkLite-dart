import 'dart:math';
import 'package:sklite/base.dart';
import 'package:sklite/utils/mathutils.dart';
import 'package:sklite/utils/exceptions.dart';

/// An implementation of sklearn.svm.SVC.
/// ---------------
///
/// https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
class SVC extends Classifier {
  List<dynamic> supportVectors;
  List<dynamic> dualCoef;
  List<dynamic> intercept;
  List<dynamic> nSupport;
  List<int> classes;
  String kernel;
  double gamma;
  double coef0;
  int degree;
  final List<String> _supported = ["rbf", "linear", "sigmoid", "poly"];

  /// To manually instantiate the SVC. The parameters
  /// are lifted directly from scikit-learn.
  /// See the attributes here:
  /// https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
  SVC(this.supportVectors, this.dualCoef, this.intercept, this.nSupport,
      this.classes, this.kernel, this.gamma, this.coef0, this.degree);

  factory SVC.fromMap(Map params) {
    return SVC(
        params["support_vectors_"],
        params["dual_coef_"],
        params["_intercept_"],
        params["n_support_"],
        List<int>.from(params["classes_"]),
        params["kernel"],
        params["_gamma"],
        params["coef0"],
        params["degree"]);
  }

  /// Implementation of sklearn.svm.SVC.predict.
  @override
  int predict(List<double> X) {
    List<double> kernels;
    List<int> starts = List<int>.filled(classes.length, 0);
    List<int> ends = List(classes.length);
    if (kernel == "rbf")
      kernels = rbf(X);
    else if (kernel == "linear")
      kernels = linear(X);
    else if (kernel == "sigmoid")
      kernels = sigmoid(X);
    else if (kernel == "poly")
      kernels = sigmoid(X);
    else
      throw InvalidSVMKernelException(
          "Unsupported kernel $kernel, supported are: ${_supported.join(", ")}");

    for (int i = 1; i < classes.length; i++) {
      int start = 0;
      for (int j = 0; j < i; j++) {
        start += nSupport[j];
      }
      starts[i] = start;
    }
    classes.asMap().forEach((i, v) => ends[i] = nSupport[i] + starts[i]);

    if (classes.length == 2) {
      for (int i = 0; i < kernels.length; i++) {
        kernels[i] = -kernels[i];
      }

      double decision = 0.0;
      for (int k = starts[1]; k < ends[1]; k++) {
        decision += kernels[k] * dualCoef[0][k];
      }
      for (int k = starts[0]; k < ends[0]; k++) {
        decision += kernels[k] * dualCoef[0][k];
      }
      decision += intercept[0];
    }

    List<double> decisions = List(intercept.length);
    for (int i = 0, d = 0, l = classes.length; i < l; i++) {
      for (int j = i + 1; j < l; j++) {
        double tmp = 0.0;
        for (int k = starts[j]; k < ends[j]; k++) {
          tmp += dualCoef[i][k] * kernels[k];
        }
        for (int k = starts[i]; k < ends[i]; k++) {
          tmp += dualCoef[j - 1][k] * kernels[k];
        }
        decisions[d] = tmp + intercept[d];
        d++;
      }
    }

    List<int> votes = List(intercept.length);
    for (int i = 0, d = 0, l = classes.length; i < l; i++) {
      for (int j = i + 1; j < l; j++) {
        votes[d] = decisions[d] > 0 ? i : j;
        d++;
      }
    }
    List<int> amounts = List<int>.filled(classes.length, 0);
    votes.asMap().forEach((i, v) => amounts[votes[i]] += 1);

    int classVal = -1;
    int idx = -1;
    for (int i = 0, l = amounts.length; i < l; i++) {
      if (amounts[i] > classVal) {
        classVal = amounts[i];
        idx = i;
      }
    }

    return classes[idx];
  }

  /// RBF kernel.
  /// ---------------
  ///
  /// K(x, y) = exp(-gamma ||x-y||^2)
  List<double> rbf(List<double> X) {
    List<double> kernels = List(supportVectors.length);
    for (int i = 0; i < supportVectors.length; i++) {
      double kernel = 0.0;
      for (int j = 0; j < supportVectors[i].length; j++) {
        kernel += pow(supportVectors[i][j] - X[j], 2);
      }
      kernels[i] = exp(this.gamma * -1.0 * kernel);
    }
    return kernels;
  }

  /// Linear kernel.
  /// ---------------
  ///
  /// K(x, y) = <f(x), f(y)>
  List<double> linear(List<double> X) {
    List<double> kernels = List(supportVectors.length);
    for (int i = 0; i < supportVectors.length; i++) {
      double kernel = 0.0;
      for (int j = 0; j < supportVectors[i].length; j++) {
        kernel += supportVectors[i][j] * X[j];
      }
      kernels[i] = kernel;
    }
    return kernels;
  }

  /// Sigmoid kernel.
  /// ---------------
  ///
  /// K(X, Y) = tanh([gamma] <X, Y> + [coef0])
  List<double> sigmoid(List<double> X) {
    List<double> kernels = List(supportVectors.length);
    for (int i = 0; i < supportVectors.length; i++) {
      double kernel = 0.0;
      for (int j = 0; j < supportVectors[i].length; j++) {
        kernel += supportVectors[i][j] * X[j];
      }
      kernels[i] = tanh((this.gamma * kernel) + this.coef0);
    }
    return kernels;
  }

  /// Polynomial kernel.
  /// ---------------
  ///
  /// K(X, Y) = ([gamma] <X, Y> + [coef0])^[degree]
  List<double> poly(List<double> X) {
    List<double> kernels = List(supportVectors.length);
    for (int i = 0; i < supportVectors.length; i++) {
      double kernel = 0.0;
      for (int j = 0; j < supportVectors[i].length; j++) {
        kernel += supportVectors[i][j] * X[j];
      }
      kernels[i] = pow((this.gamma * kernel) + this.coef0, this.degree);
    }
    return kernels;
  }
}

/// An implementation of sklearn.svm.SVC.
/// ---------------
///
/// https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
class LinearSVC extends Classifier {
  List<dynamic> coef;
  List<dynamic> intercept;
  List<dynamic> classes;

  /// To manually instantiate the LinearSVC. The parameters
  /// are lifted directly from scikit-learn.
  /// See the attributes here:
  /// https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  LinearSVC(this.coef, this.intercept, this.classes);

  factory LinearSVC.fromMap(Map<String, dynamic> params) {
    return LinearSVC(params["coef_"], params["intercept_"], params["classes_"]);
  }

  /// Implementation of sklearn.svm.LinearSVC.predict.
  @override
  int predict(List<double> X) {
    if (classes.length == 2) return _binPredict(X);
    return _predict(X);
  }

  int _predict(List<double> X) {
    int idx = 0;
    double classVal = 0.0;
    for (int i = 0, il = intercept.length; i < il; i++) {
      double prob = 0.0;
      for (int j = 0, jl = coef[0].length; j < jl; j++) {
        prob += coef[i][j] * X[j];
      }
      if (prob + intercept[i] > classVal) {
        classVal = prob + intercept[i];
        idx = i;
      }
    }
    return classes[idx];
  }

  int _binPredict(List<double> X) {
    double prob = 0.0;
    coef.asMap().forEach((i, v) => prob += coef[i] * X[i]);
    if (prob + intercept[0] > 0.0) {
      return classes[1];
    }
    return classes[0];
  }
}
