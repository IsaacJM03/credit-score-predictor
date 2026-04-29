import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/applicant_input.dart';
import '../models/loan_decision_result.dart';
import '../models/post_loan_result.dart';

/// Default Hugging Face Space API base (HTTPS, no trailing slash).
/// Change in-app via settings or override here for local/dev.
const String kDefaultLoanlyBaseUrl =
    'https://isaacmwesigwa-credit-score-predictor-models.hf.space';

class LoanlyApiException implements Exception {
  LoanlyApiException(this.message, {this.statusCode});
  final String message;
  final int? statusCode;

  @override
  String toString() => 'LoanlyApiException($statusCode): $message';
}

class LoanlyApiService {
  LoanlyApiService({String? baseUrl})
      : baseUrl = (baseUrl ?? kDefaultLoanlyBaseUrl).replaceAll(RegExp(r'/+$'), '');

  final String baseUrl;

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Future<LoanDecisionResult> loanDecision(ApplicantInput input) async {
    final res = await http.post(
      _uri('/predict/loan-decision'),
      headers: const {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: jsonEncode(input.toLoanDecisionJson()),
    );
    _throwIfBad(res);
    final map = jsonDecode(res.body) as Map<String, dynamic>;
    return LoanDecisionResult.fromJson(map);
  }

  Future<PostLoanResult> postLoanMonitoring(ApplicantInput input) async {
    final res = await http.post(
      _uri('/predict/post-loan-monitoring'),
      headers: const {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: jsonEncode(input.toPostLoanMonitoringJson()),
    );
    _throwIfBad(res);
    final map = jsonDecode(res.body) as Map<String, dynamic>;
    return PostLoanResult.fromJson(map);
  }

  void _throwIfBad(http.Response res) {
    if (res.statusCode >= 200 && res.statusCode < 300) return;
    String detail = res.body;
    try {
      final j = jsonDecode(res.body);
      if (j is Map<String, dynamic> && j['detail'] != null) {
        final d = j['detail'];
        detail = d is String ? d : jsonEncode(d);
      }
    } catch (_) {}
    throw LoanlyApiException(detail.trim().isEmpty ? 'Request failed' : detail,
        statusCode: res.statusCode);
  }
}
