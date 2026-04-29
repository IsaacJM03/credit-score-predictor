class PostLoanResult {
  PostLoanResult({
    required this.status,
    required this.statusChip,
    required this.recommendedAction,
    required this.notes,
    required this.combinedPd,
    required this.modelPd,
    required this.behaviouralPd,
    required this.riskScore,
    required this.missedPayments,
    required this.monthsSinceDisbursement,
    required this.principalRepaidRatio,
  });

  final String status;
  final String statusChip;
  final String recommendedAction;
  final List<String> notes;
  final double combinedPd;
  final double modelPd;
  final double behaviouralPd;
  final double riskScore;
  final int missedPayments;
  final int monthsSinceDisbursement;
  final double principalRepaidRatio;

  factory PostLoanResult.fromJson(Map<String, dynamic> json) {
    final s4 = json['stage_4_health_output'] as Map<String, dynamic>? ?? {};
    final s5 = json['stage_5_action_plan'] as Map<String, dynamic>? ?? {};

    return PostLoanResult(
      status: (s5['status'] as String?) ?? '—',
      statusChip: (s5['status_chip'] as String?) ?? 'amber',
      recommendedAction: (s5['recommended_action'] as String?) ?? '',
      notes: List<String>.from(s5['notes'] as List? ?? []),
      combinedPd: _toDouble(s4['combined_default_probability']),
      modelPd: _toDouble(s4['model_default_probability']),
      behaviouralPd: _toDouble(s4['behavioural_default_probability']),
      riskScore: _toDouble(s4['risk_score']),
      missedPayments: (s4['missed_payments'] as num?)?.toInt() ?? 0,
      monthsSinceDisbursement:
          (s4['months_since_disbursement'] as num?)?.toInt() ?? 0,
      principalRepaidRatio: _toDouble(s4['principal_repaid_ratio']),
    );
  }

  static double _toDouble(dynamic v) {
    if (v == null) return 0;
    if (v is num) return v.toDouble();
    return double.tryParse('$v') ?? 0;
  }
}
