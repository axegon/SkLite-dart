import 'package:sklite/tree/tree.dart';
import 'package:sklite/naivebayes/naive_bayes.dart';
import 'package:sklite/SVM/SVM.dart';
import 'package:sklite/neighbors/neighbors.dart';
import 'package:sklite/ensemble/forest.dart';
import 'package:sklite/neural_network/neural_network.dart';
import 'dart:io';
import 'dart:convert';

/// For development only.
String readFile(String path){
  return File(path).readAsStringSync();
}

/// For development only.
Map<String, dynamic> readJsonFile(String path) {
  return json.decode(readFile(path));
}

void main(){
  List<double> X = [5.0, 2.0, 3.5, 1.0];
  print("DecisionTreeClassifier");
  var params0 = readJsonFile('models/dt.json');
  DecisionTreeClassifier x = DecisionTreeClassifier.fromMap(params0);
  print(x.predict(X));
  print('------------------------------------');
  print("GaussianNB");
  var params1 = readJsonFile('models/nbc.json');
  GaussianNB n = GaussianNB.fromMap(params1);
  print(n.predict(X));
  print('------------------------------------');
  print("SVC");
  var params2 = readJsonFile('models/svc.json');
  SVC s = SVC.fromMap(params2);
  print(s.predict(X));
  print('------------------------------------');
  print("LinearSVC");
  var params3 = readJsonFile('models/linearsvc.json');
  LinearSVC l = LinearSVC.fromMap(params3);
  print(l.predict(X));
  print('------------------------------------');
  print("BernoulliNB");
  var params4 = readJsonFile('models/bernoullinb.json');
  BernoulliNB b = BernoulliNB.fromMap(params4);
  print(b.predict(X));
  print('------------------------------------');
  print("KNeighborsClassifier");
  var params5 = readJsonFile("models/kneighbors.json");
  KNeighborsClassifier k = KNeighborsClassifier.fromMap(params5);
  print(k.predict(X));
  print('------------------------------------');
  print("RandomForestClassifier");
  var params6 = readJsonFile("models/randomforest.json");
  RandomForestClassifier r = RandomForestClassifier.fromMap(params6);
  print(r.predict(X));
  print('------------------------------------');
  print("MLPClassifier");
  var params7 = readJsonFile("models/mlpclassifier.json");
  MLPClassifier m = MLPClassifier.fromMap(params7);
  print(m.predict(X));
}