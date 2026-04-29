/// Mirrors backend `ApplicantRequest` / `PostLoanRequest` field names for JSON.
class ApplicantInput {
  const ApplicantInput({
    required this.creditScore,
    required this.monthlyIncome,
    required this.loanAmount,
    required this.existingDebts,
    required this.age,
    required this.employmentStatus,
    this.numPreviousLoans = 0,
    this.paymentHistory = 'good',
    this.monthsSinceDisbursement = 0,
    this.currentLoanBalance,
    this.missedPayments = 0,
  });

  final double creditScore;
  final double monthlyIncome;
  final double loanAmount;
  final double existingDebts;
  final int age;
  final String employmentStatus;
  final int numPreviousLoans;
  final String paymentHistory;

  /// Post-loan only
  final int monthsSinceDisbursement;
  final double? currentLoanBalance;
  final int missedPayments;

  Map<String, dynamic> toLoanDecisionJson() => {
        'credit_score': creditScore,
        'monthly_income': monthlyIncome,
        'loan_amount': loanAmount,
        'existing_debts': existingDebts,
        'age': age,
        'employment_status': employmentStatus,
        'num_previous_loans': numPreviousLoans,
        'payment_history': paymentHistory,
      };

  Map<String, dynamic> toPostLoanMonitoringJson() => {
        ...toLoanDecisionJson(),
        'months_since_disbursement': monthsSinceDisbursement,
        'current_loan_balance': currentLoanBalance,
        'missed_payments': missedPayments,
      };

  ApplicantInput copyWith({
    double? creditScore,
    double? monthlyIncome,
    double? loanAmount,
    double? existingDebts,
    int? age,
    String? employmentStatus,
    int? numPreviousLoans,
    String? paymentHistory,
    int? monthsSinceDisbursement,
    double? currentLoanBalance,
    int? missedPayments,
  }) {
    return ApplicantInput(
      creditScore: creditScore ?? this.creditScore,
      monthlyIncome: monthlyIncome ?? this.monthlyIncome,
      loanAmount: loanAmount ?? this.loanAmount,
      existingDebts: existingDebts ?? this.existingDebts,
      age: age ?? this.age,
      employmentStatus: employmentStatus ?? this.employmentStatus,
      numPreviousLoans: numPreviousLoans ?? this.numPreviousLoans,
      paymentHistory: paymentHistory ?? this.paymentHistory,
      monthsSinceDisbursement:
          monthsSinceDisbursement ?? this.monthsSinceDisbursement,
      currentLoanBalance: currentLoanBalance ?? this.currentLoanBalance,
      missedPayments: missedPayments ?? this.missedPayments,
    );
  }
}
