import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/applicant_input.dart';
import '../theme/loanly_theme.dart';

/// Shared applicant fields used by Loan Approval & Post-Loan Monitoring.
class ApplicantFieldsController {
  ApplicantFieldsController({
    ApplicantInput initial = const ApplicantInput(
      creditScore: 680,
      monthlyIncome: 5500,
      loanAmount: 20000,
      existingDebts: 800,
      age: 34,
      employmentStatus: 'employed',
      numPreviousLoans: 2,
      paymentHistory: 'good',
    ),
  })  : creditScore = TextEditingController(text: initial.creditScore.toStringAsFixed(0)),
        monthlyIncome = TextEditingController(text: initial.monthlyIncome.toStringAsFixed(0)),
        loanAmount = TextEditingController(text: initial.loanAmount.toStringAsFixed(0)),
        existingDebts = TextEditingController(text: initial.existingDebts.toStringAsFixed(0)),
        age = TextEditingController(text: '${initial.age}'),
        numPreviousLoans = TextEditingController(text: '${initial.numPreviousLoans}'),
        employmentStatus = initial.employmentStatus,
        paymentHistory = initial.paymentHistory;

  final TextEditingController creditScore;
  final TextEditingController monthlyIncome;
  final TextEditingController loanAmount;
  final TextEditingController existingDebts;
  final TextEditingController age;
  final TextEditingController numPreviousLoans;

  String employmentStatus;
  String paymentHistory;

  void dispose() {
    creditScore.dispose();
    monthlyIncome.dispose();
    loanAmount.dispose();
    existingDebts.dispose();
    age.dispose();
    numPreviousLoans.dispose();
  }

  ApplicantInput toApplicantInput({
    int monthsSinceDisbursement = 0,
    double? currentLoanBalance,
    int missedPayments = 0,
  }) {
    return ApplicantInput(
      creditScore: double.tryParse(creditScore.text.trim()) ?? 650,
      monthlyIncome: double.tryParse(monthlyIncome.text.trim()) ?? 0,
      loanAmount: double.tryParse(loanAmount.text.trim()) ?? 0,
      existingDebts: double.tryParse(existingDebts.text.trim()) ?? 0,
      age: int.tryParse(age.text.trim()) ?? 25,
      employmentStatus: employmentStatus,
      numPreviousLoans: int.tryParse(numPreviousLoans.text.trim()) ?? 0,
      paymentHistory: paymentHistory,
      monthsSinceDisbursement: monthsSinceDisbursement,
      currentLoanBalance: currentLoanBalance,
      missedPayments: missedPayments,
    );
  }
}

class ApplicantFieldsForm extends StatefulWidget {
  const ApplicantFieldsForm({
    super.key,
    required this.controller,
    this.showLoanSlider = true,
  });

  final ApplicantFieldsController controller;
  final bool showLoanSlider;

  static const employmentOptions = [
    ('employed', 'Employed'),
    ('self_employed', 'Self-employed'),
    ('unemployed', 'Unemployed'),
    ('student', 'Student'),
    ('retired', 'Retired'),
  ];

  static const paymentOptions = [
    ('poor', 'Poor'),
    ('fair', 'Fair'),
    ('good', 'Good'),
    ('excellent', 'Excellent'),
  ];

  @override
  State<ApplicantFieldsForm> createState() => _ApplicantFieldsFormState();
}

class _ApplicantFieldsFormState extends State<ApplicantFieldsForm> {
  late double _creditSliderValue;

  @override
  void initState() {
    super.initState();
    final raw = double.tryParse(widget.controller.creditScore.text) ?? 680;
    _creditSliderValue = raw.clamp(300, 900);
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          'Applicant profile',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w700,
                color: LoanlyTheme.accent,
              ),
        ),
        const SizedBox(height: 16),
        if (widget.showLoanSlider) ...[
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Credit score',
                style: TextStyle(color: LoanlyTheme.textMuted),
              ),
              Text(
                _creditSliderValue.round().toString(),
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: cs.primary,
                  fontSize: 16,
                ),
              ),
            ],
          ),
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              activeTrackColor: LoanlyTheme.accent,
              inactiveTrackColor: LoanlyTheme.border,
              thumbColor: LoanlyTheme.accent,
              overlayColor: LoanlyTheme.accent.withValues(alpha: 0.2),
            ),
            child: Slider(
              min: 300,
              max: 900,
              divisions: 60,
              value: _creditSliderValue,
              onChanged: (v) {
                setState(() => _creditSliderValue = v);
                widget.controller.creditScore.text = v.round().toString();
              },
            ),
          ),
          const SizedBox(height: 8),
        ] else
          TextFormField(
            controller: widget.controller.creditScore,
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
            decoration: const InputDecoration(
              labelText: 'Credit score',
              hintText: '300–900',
            ),
          ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: TextFormField(
                controller: widget.controller.monthlyIncome,
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                decoration: const InputDecoration(
                  labelText: 'Monthly income',
                  prefixText: '\$ ',
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: TextFormField(
                controller: widget.controller.loanAmount,
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                decoration: const InputDecoration(
                  labelText: 'Loan amount',
                  prefixText: '\$ ',
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: DropdownButtonFormField<String>(
                value: widget.controller.employmentStatus,
                decoration: const InputDecoration(labelText: 'Employment'),
                items: ApplicantFieldsForm.employmentOptions
                    .map((e) => DropdownMenuItem(value: e.$1, child: Text(e.$2)))
                    .toList(),
                onChanged: (v) {
                  setState(() {
                    if (v != null) widget.controller.employmentStatus = v;
                  });
                },
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: DropdownButtonFormField<String>(
                value: widget.controller.paymentHistory,
                decoration: const InputDecoration(labelText: 'Payment history'),
                items: ApplicantFieldsForm.paymentOptions
                    .map((e) => DropdownMenuItem(value: e.$1, child: Text(e.$2)))
                    .toList(),
                onChanged: (v) {
                  setState(() {
                    if (v != null) widget.controller.paymentHistory = v;
                  });
                },
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: TextFormField(
                controller: widget.controller.existingDebts,
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                decoration: const InputDecoration(
                  labelText: 'Existing debts / month',
                  prefixText: '\$ ',
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: TextFormField(
                controller: widget.controller.age,
                keyboardType: TextInputType.number,
                inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                decoration: const InputDecoration(labelText: 'Age'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        TextFormField(
          controller: widget.controller.numPreviousLoans,
          keyboardType: TextInputType.number,
          inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          decoration: const InputDecoration(
            labelText: 'Previous loans (count)',
            hintText: 'Optional — affects velocity signal',
          ),
        ),
      ],
    );
  }
}
