class InvalidSVMKernelException implements Exception {
  String cause;

  InvalidSVMKernelException(this.cause) {
    print(cause);
  }
}

class InvalidActivationException implements Exception {
  String cause;

  InvalidActivationException(this.cause) {
    print(cause);
  }
}

class InvalidLoggingLevelException implements Exception {
  String cause;

  InvalidLoggingLevelException(this.cause) {
    print(cause);
  }
}
