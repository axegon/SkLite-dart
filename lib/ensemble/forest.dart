import 'package:sklite/utils/mathutils.dart';
import 'package:sklite/base.dart';
import 'package:sklite/tree/tree.dart';


/// An implementation of sklearn.ensemble.RandomForestClassifier
/// ---------------
///
/// https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
class RandomForestClassifier extends Classifier {
  List<int> classes;
  List<DecisionTreeClassifier> _dtrees = [];
  List<dynamic> dtrees;


  /// To manually instantiate the RandomForestClassifier. The parameters
  /// are lifted directly from scikit-learn.
  /// See the attributes here:
  /// https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  RandomForestClassifier(this.classes, this.dtrees){
    initDtrees(dtrees);
  }

  /// Override from Classifier.
  factory RandomForestClassifier.fromMap(Map<String, dynamic> params) {
    return RandomForestClassifier(List<int>.from(params["classes_"]), params["dtrees"]);
  }

  /// Initializes the decision [trees] within the forest.
  /// Each of those instantiates a DecisionTreeClassifier.
  void initDtrees(List<dynamic> trees) {
    if (_dtrees.length > 0) return null;
    for (int i = 0; i < trees.length; i++) {
      trees[i]["classes_"] = classes;
      _dtrees.add(DecisionTreeClassifier.fromMap(trees[i]));
    }
  }

  /// Implementation of sklearn.ensemble.RandomForestClassifier.predict.
  @override
  int predict(List<double> X) {
    var cls = List<dynamic>.filled(_dtrees[0].classes.length, 0);
    _dtrees.asMap().forEach((i, v) => cls[_dtrees[i].predict(X)]++);
    return classes[argmax(cls)];
  }
}
