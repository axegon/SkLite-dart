import 'dart:math';
import 'package:sklite/base.dart';
import 'package:sklite/utils/mathutils.dart';

/// An implementation of sklearn.naive_bayes.KNeighborsClassifier
/// ---------------
///
/// https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
class GaussianNB extends Classifier {
  List<double> classPrior;
  List<List<dynamic>> sigma;
  List<List<dynamic>> theta;
  List<int> classes;

  /// To manually instantiate the GaussianNB. The parameters
  /// are lifted directly from scikit-learn.
  /// See the attributes here:
  /// https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
  GaussianNB(this.classPrior, this.sigma, this.theta, this.classes);

  factory GaussianNB.fromMap(Map<String, dynamic> params) {
    return GaussianNB(
        List<double>.from(params['class_prior_']),
        List<List<dynamic>>.from(params['sigma_']),
        List<List<dynamic>>.from(params['theta_']),
        List<int>.from(params["classes_"]));
  }

  /// Implementation of sklearn.nayve_bayes.GaussianNB.predict.
  @override
  int predict(List<double> X) {
    // List<double> probabilities = List(sigma.length);
    List<double> probabilities =
        List<double>.generate(sigma.length, (index) => index.toDouble());
    for (int i = 0; i < sigma.length; i++) {
      double sum = 0.0;
      for (int j = 0; j < sigma[0].length; j++) {
        sum += log(2.0 * pi * sigma[i][j]);
      }
      double nij = -0.5 * sum;
      sum = 0.0;
      for (int j = 0; j < sigma.length; j++) {
        sum += pow(X[j] - theta[i][j], 2.0) / sigma[i][j];
      }
      nij -= 0.5 * sum;
      probabilities[i] = log(classPrior[i]) + nij;
    }

    return classes[argmax(probabilities)];
  }
}

/// An implementation of sklearn.naive_bayes.BernoulliNB
/// ---------------
///
/// https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
class BernoulliNB extends Classifier {
  List<double> priors;
  List<List<dynamic>> negProbs;
  List<List<dynamic>> delProbes;
  List<int> classes;

  /// To manually instantiate the GaussianNB. The parameters
  /// are lifted directly from scikit-learn.
  /// See the attributes here:
  /// https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
  BernoulliNB(this.priors, this.negProbs, this.delProbes, this.classes);

  factory BernoulliNB.fromMap(Map<String, dynamic> params) {
    return BernoulliNB(
        List<double>.from(params["class_log_prior_"]),
        List<List<dynamic>>.from(params["neg_prob_"]),
        List<List<dynamic>>.from(params["delta_probs_"]),
        List<int>.from(params["classes_"]));
  }

  /// Implementation of sklearn.nayve_bayes.BernoulliNB.predict.
  @override
  int predict(List<double> X) {
    int nClasses = classes.length;
    int nFeatures = delProbes.length;

    var jll = List<double>.generate(nClasses, (index) => index.toDouble());
    for (int i = 0; i < nClasses; i++) {
      double sum = 0.0;
      for (int j = 0; j < nFeatures; j++) {
        sum += X[i] * delProbes[j][i];
      }
      jll[i] = sum;
    }
    for (int i = 0; i < nClasses; i++) {
      double sum = 0.0;
      for (int j = 0; j < nFeatures; j++) {
        sum += negProbs[i][j];
      }
      jll[i] += priors[i] + sum;
    }
    var idx = 0;

    classes.asMap().forEach((i, v) => idx = jll[i] > jll[idx] ? i : idx);
    return classes[idx];
  }
}
