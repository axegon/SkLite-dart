import 'package:sklite/base.dart';
import 'package:sklite/utils/mathutils.dart';

/// An implementation of sklearn.tree.DecisionTreeClassifier.
/// ---------------
///
/// https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
class DecisionTreeClassifier extends Classifier {
  List<int> childrenLeft;
  List<int> childrenRight;
  List<double> threshold;
  List<int> features;
  List<List<dynamic>> values;
  List<int> classes;

  /// To manually instantiate the DecisionTreeClassifier. The parameters
  /// are lifted directly from scikit-learn.
  /// See the attributes here:
  /// https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
  DecisionTreeClassifier(this.childrenLeft, this.childrenRight, this.threshold,
      this.features, this.values, this.classes);

  factory DecisionTreeClassifier.fromMap(Map params) {
    return DecisionTreeClassifier(
        List<int>.from(params["children_left"]),
        List<int>.from(params["children_right"]),
        List<double>.from(params["threshold"]),
        List<int>.from(params["feature"]),
        List<List<dynamic>>.from(params["value"]),
        List<int>.from(params["classes_"] ?? []));
  }

  /// Implementation of sklearn.tree.DecisionTreeClassifier.predict.
  @override
  int predict(List<double> X) {
    return _predict(X);
  }

  int _predict(List<double> X, [int? node]) {
    node ??= 0;
    if (threshold[node] != -2) {
      if (X[features[node]] <= threshold[node])
        return _predict(X, childrenLeft[node]);
      return _predict(X, childrenRight[node]);
    }
    return classes[argmax(List<double>.from(values[node]))];
  }
}
