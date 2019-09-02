import 'dart:math';
import 'package:sklite/utils/mathutils.dart' as mathutils;
import 'package:sklite/base.dart';
import 'package:sklite/utils/exceptions.dart';

/// An implementation of sklearn.neural_network.MLPClassifier
/// ---------------
///
/// https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
class MLPClassifier extends Classifier {
  List<int> layers;
  List<List<dynamic>> coefs;
  List<List<dynamic>> intercepts;
  List<int> classes;
  String activation;
  String outActivation;
  final List<String> _activations = ["logistic", "relu", "tanh", "softmax"];

  /// To manually instantiate the MLPClassifier. The parameters
  /// are lifted directly from scikit-learn.
  /// See the attributes here:
  /// https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
  MLPClassifier(this.layers, this.coefs, this.intercepts, this.classes,
      this.activation, this.outActivation);

  factory MLPClassifier.fromMap(Map params) {
    return MLPClassifier(
        List<int>.from(params["layers"]),
        List<List<dynamic>>.from(params["coefs_"]),
        List<List<dynamic>>.from(params["intercepts_"]),
        List<int>.from(params["classes_"]),
        params["activation"],
        params["out_activation"]);
  }

  /// Logistic activation function:
  /// ---------------
  ///
  /// f(x) = 1 / (1 + exp(-x))
  List<double> logistic(List<double> val) {
    val.asMap().forEach((i, v) => val[i] = 1.0 / (1.0 + exp(-val[i])));
    return val;
  }

  /// Rectified Linear unit activation function:
  /// --------------
  ///
  /// f(x) = max(0, x)
  List<double> relu(List<double> val) {
    val.asMap().forEach((i, v) => val[i] = max(0, val[i]));
    return val;
  }

  /// Hyperbolic tangent activation function:
  /// --------------
  ///
  /// f(x) = tanh(x)
  List<double> tanh(List<double> val) {
    val.asMap().forEach((i, v) => val[i] = mathutils.tanh(val[i]));
    return val;
  }

  /// Softmax activation function:
  /// ---------------
  ///
  /// \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
  /// LaTeX in the absence of a better way to express it...
  List<double> softmax(List<double> val) {
    double max = mathutils.NEGATIVE_INFINITY;
    double sum = 0.0;
    val.asMap().forEach((i, v) => max = val[i] > max ? val[i] : max);
    val.asMap().forEach((i, v) => val[i] = exp(val[i] - max));
    val.asMap().forEach((i, v) => sum += val[i]);
    val.asMap().forEach((i, v) => val[i] /= sum);
    return val;
  }

  /// No-op activation function
  /// ---------------
  ///
  /// f(x) = x
  List<double> identity(List<double> val) {
    return val;
  }

  /// Serves as a getter function for all the activation functions,
  /// as [val] being a list of doubles(within a layer of the network.
  /// [activationType] points to which activation to be used,
  /// using the activation from the constructor by default.
  List<double> _activation(List<double> val, [String activationType]) {
    activationType ??= activation;
    if (activationType == "logistix") return logistic(val);
    if (activationType == "relu") return relu(val);
    if (activationType == "tanh") return tanh(val);
    if (activationType == "softmax") return softmax(val);
    if (activationType == "identity") return identity(val);
    throw InvalidActivationException(
        "Invalid $activationType activation used, supported are  ${_activations.join(", ")}.");
  }

  /// Implementation of sklearn.neural_network.MLPClassifier.predict.
  @override
  int predict(List<double> X) {
    List<List<dynamic>> network = [X, null, null];
    layers.asMap().forEach(
        (i, v) => network[i + 1] = List<double>.filled(layers[i], 0.0));
    for (int i = 0; i < network.length - 1; i++) {
      for (int j = 0; j < network[i + 1].length; j++) {
        network[i + 1][j] = intercepts[i][j];
        network[i].asMap().forEach(
            (l, v) => network[i + 1][j] += network[i][l] * coefs[i][l][j]);
      }
      if ((i + 1) < (network.length - 1)) {
        network[i + 1] = _activation(network[i + 1]);
      }
    }
    network[network.length - 1] =
        _activation(network[network.length - 1], outActivation);

    if (network[network.length - 1].length == 1) {
      if (network[network.length - 1][0] > 0.5) {
        return classes[1];
      }
      return classes[0];
    }

    return classes[mathutils.argmax(network[network.length - 1])];
  }
}
