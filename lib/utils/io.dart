import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

/// For development only.
String readFile(String path){
  return File(path).readAsStringSync();
}

/// For development only.
Map<String, dynamic> readJsonFile(String path) {
  return json.decode(readFile(path));
}

Future<String> loadModelContext(BuildContext context, String path) async {
  return await DefaultAssetBundle.of(context)
      .loadString(path);
}

Future<String> loadModel(String path) async {
  return await rootBundle.loadString(path);
}