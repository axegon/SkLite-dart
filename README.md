# SkLite-dart

[![SkLite Demo App](http://img.youtube.com/vi/K5D1T1VJBR8/0.jpg)](http://www.youtube.com/watch?v=K5D1T1VJBR8 "SkLite-dart demo")

* Porting Scikit-Learn models to Flutter *

## Getting Started

The library uses pre-trained models using scikit-learn and [sklite](https://github.com/axegon/SkLite) for python. Once built and exported, the models can be used in Flutter with ease. You need to load them inside your application and use them accordingly to your case:

You need to setup a local static directory where you can store the generated JSON file using [sklite](https://github.com/axegon/SkLite).

Put sklite in your pubspec.yaml file, import whichever model it is you intend to use and use the `predict` method as you would in scikit-learn. Use the
`loadModel` in utils/dart.io for your application, the other functions are for development and testing only.

## Supported models

| IMPLEMENTATION                     | STATUS |
|------------------------------------|--------|
| KNeighborsClassifier               | ✓      |
| SVC                                | ✓      |
| GaussianProcessClassifier          |        |
| DecisionTreeClassifier             | ✓      |
| RandomForestClassifier             | ✓      |
| MLPClassifier                      | ✓      |
| AdaBoostClassifier                 |        |
| GaussianNB                         | ✓      |
| QuadraticDiscriminantAnalysis      |        |
| BernoulliNB                        | ✓      |
| LinearSVC                          | ✓      |

## Usage

You first need to add the library to your project's pubspec.yaml. For the time being the library hasn't been published to pub.dev (should be soon enough though).

```
dependencies:
  sklite: ^0.0.1
```


Add a static assets directory inside your Flutter project:

```
flutter:
  uses-material-design: true
  assets:
    - assets/
```

Put the generated model using [sklite for python](https://github.com/axegon/SkLite) in the specified directory

Import the library in your application and load the model:

```
import 'package:flutter/material.dart';
import 'package:sklite/SVM/SVM.dart';
import 'package:sklite/utils/io.dart';
import 'dart:convert';

void main() => runApp(new MaterialApp(
  home: new HomePage(),
  debugShowCheckedModeBanner: false)
);

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() {
    return new _HomePageState();
  }
}

class _HomePageState extends State<HomePage> {
  SVC svc;

  _HomePageState() {
    loadModel("assets/svcmnist.json").then((x) {
      this.svc = SVC.fromMap(json.decode(x));
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        // add any widget with svc.predict() callback
    );
  }
```
