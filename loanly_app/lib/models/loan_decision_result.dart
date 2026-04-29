class LoanDecisionResult {
  LoanDecisionResult({
    required this.stage5Decision,
    required this.reasons,
    required this.repaymentPrediction,
    required this.riskScore,
    required this.riskBand,
    required this.probabilityOfDefault,
    required this.postLoanDefaultProbability,
    required this.classifierScores,
    required this.postLoanScores,
    required this.fraudFlagged,
    required this.segmentLabel,
    required this.derivedDti,
    required this.derivedLti,
  });

  final String stage5Decision;
  final List<String> reasons;
  final String repaymentPrediction;
  final double riskScore;
  final String riskBand;
  final double probabilityOfDefault;
  final double? postLoanDefaultProbability;
  final Map<String, double> classifierScores;
  final Map<String, double> postLoanScores;
  final bool fraudFlagged;
  final String? segmentLabel;
  final double derivedDti;
  final double derivedLti;

  factory LoanDecisionResult.fromJson(Map<String, dynamic> json) {
    final s4 =
        json['stage_4_decision_output'] as Map<String, dynamic>? ?? {};
    final s5 = json['stage_5_lending_dashboard'] as Map<String, dynamic>? ?? {};
    final s3 = json['stage_3_ml_engine'] as Map<String, dynamic>? ?? {};
    final s2 = json['stage_2_preprocessing'] as Map<String, dynamic>? ?? {};
    final derived = s2['derived'] as Map<String, dynamic>? ?? {};

    final fraud = s4['fraud'] as Map<String, dynamic>? ?? {};
    final seg = s4['segmentation'];
    String? segLabel;
    if (seg is Map<String, dynamic>) {
      segLabel = seg['label'] as String?;
    }

    final clsRaw = s3['classifier_scores'];
    final postRaw = s3['post_loan_scores'];

    return LoanDecisionResult(
      stage5Decision: (s5['final_decision'] as String?) ?? '—',
      reasons: List<String>.from(s5['reasons'] as List? ?? []),
      repaymentPrediction: (s4['repayment_prediction'] as String?) ?? '—',
      riskScore: _toDouble(s4['risk_score']),
      riskBand: (s4['risk_band'] as String?) ?? '—',
      probabilityOfDefault: _toDouble(s4['probability_of_default']),
      postLoanDefaultProbability: s4['post_loan_default_probability'] != null
          ? _toDouble(s4['post_loan_default_probability'])
          : null,
      classifierScores: _stringDoubleMap(clsRaw),
      postLoanScores: _stringDoubleMap(postRaw),
      fraudFlagged: fraud['flagged'] == true,
      segmentLabel: segLabel,
      derivedDti: _toDouble(derived['debt_to_income_ratio']),
      derivedLti: _toDouble(derived['loan_to_income_ratio']),
    );
  }

  static double _toDouble(dynamic v) {
    if (v == null) return 0;
    if (v is num) return v.toDouble();
    return double.tryParse('$v') ?? 0;
  }

  static Map<String, double> _stringDoubleMap(dynamic raw) {
    if (raw is! Map) return {};
    final out = <String, double>{};
    raw.forEach((k, v) {
      if (v != null) out['$k'] = _toDouble(v);
    });
    return out;
  }
}
