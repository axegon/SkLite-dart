/// Abstract classifier class: all classifiers extend this class
/// since they need to have individual implementation.
/// The methods to be implemented are:
///
/// int predict(List<double> X)
///   A dart implementation of the predict method
///   in scikit-learn for the given class
///
/// void fromMap(Map<String, dynamic> params)
///   Loading the parameters from the JSON
///   file or URL into the class.
abstract class Classifier {

  /// Override in all classes extending Classifier.
  int predict(List<double> X);

  /// Override in all classes extending Classifier.

  /// Load a model from a URL using an HTTP request
  /// @TODO
  fromURL(String url) {}
}
