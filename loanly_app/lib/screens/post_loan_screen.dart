import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/post_loan_result.dart';
import '../services/loanly_api.dart';
import '../theme/loanly_theme.dart';
import '../util/format_pct.dart';
import '../widgets/applicant_fields_form.dart';

class PostLoanScreen extends StatefulWidget {
  const PostLoanScreen({super.key, required this.api});

  final LoanlyApiService api;

  @override
  State<PostLoanScreen> createState() => _PostLoanScreenState();
}

class _PostLoanScreenState extends State<PostLoanScreen> {
  late final ApplicantFieldsController _fields;
  final _balanceCtl = TextEditingController();

  double _months = 12;
  double _missed = 0;

  PostLoanResult? _result;
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _fields = ApplicantFieldsController();
  }

  @override
  void dispose() {
    _fields.dispose();
    _balanceCtl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    FocusScope.of(context).unfocus();
    if (_loading) return;
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final balText = _balanceCtl.text.trim();
      final balance =
          balText.isEmpty ? null : double.tryParse(balText.replaceAll(',', ''));

      final input = _fields.toApplicantInput(
        monthsSinceDisbursement: _months.round(),
        currentLoanBalance: balance,
        missedPayments: _missed.round(),
      );

      final out = await widget.api.postLoanMonitoring(input);
      if (!mounted) return;
      setState(() {
        _result = out;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Color _chipColor(String chip) {
    switch (chip.toLowerCase()) {
      case 'green':
        return const Color(0xFF34D399);
      case 'amber':
        return LoanlyTheme.accent;
      case 'red':
        return const Color(0xFFF87171);
      default:
        return LoanlyTheme.textMuted;
    }
  }

  @override
  Widget build(BuildContext context) {
    return CustomScrollView(
      physics: const BouncingScrollPhysics(),
      slivers: [
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(20, 12, 20, 120),
          sliver: SliverList(
            delegate: SliverChildListDelegate([
              Container(
                padding: const EdgeInsets.all(22),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  gradient: LinearGradient(
                    colors: [
                      LoanlyTheme.surfaceElevated,
                      LoanlyTheme.surface.withValues(alpha: 0.92),
                    ],
                  ),
                  border: Border.all(color: LoanlyTheme.border),
                ),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: LoanlyTheme.accent.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(
                        Icons.monitor_heart_rounded,
                        color: LoanlyTheme.accent,
                        size: 28,
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Post-loan monitoring',
                            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                  fontWeight: FontWeight.w800,
                                  color: LoanlyTheme.accent,
                                ),
                          ),
                          Text(
                            'Behaviour + LightGBM/XGBoost ensemble — aligned with `/predict/post-loan-monitoring`.',
                            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                  color: LoanlyTheme.textMuted,
                                  height: 1.35,
                                ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 18),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: ApplicantFieldsForm(controller: _fields),
                ),
              ),
              const SizedBox(height: 14),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Text(
                        'Loan servicing context',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w700,
                              color: LoanlyTheme.accent,
                            ),
                      ),
                      const SizedBox(height: 18),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            'Months since disbursement',
                            style: TextStyle(color: LoanlyTheme.textMuted),
                          ),
                          Text(
                            '${_months.round()} mo',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              color: LoanlyTheme.accent,
                            ),
                          ),
                        ],
                      ),
                      SliderTheme(
                        data: SliderTheme.of(context).copyWith(
                          activeTrackColor: LoanlyTheme.accent,
                          inactiveTrackColor: LoanlyTheme.border,
                          thumbColor: LoanlyTheme.accent,
                        ),
                        child: Slider(
                          min: 0,
                          max: 120,
                          divisions: 120,
                          value: _months.clamp(0, 120),
                          onChanged: (v) => setState(() => _months = v),
                        ),
                      ),
                      TextFormField(
                        controller: _balanceCtl,
                        keyboardType:
                            const TextInputType.numberWithOptions(decimal: true),
                        inputFormatters: [
                          FilteringTextInputFormatter.allow(RegExp(r'[\d.,]')),
                        ],
                        decoration: const InputDecoration(
                          labelText: 'Current loan balance',
                          hintText: 'Blank = original principal',
                          prefixText: '\$ ',
                        ),
                      ),
                      const SizedBox(height: 12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            'Missed payments (last 12 months)',
                            style: TextStyle(color: LoanlyTheme.textMuted),
                          ),
                          Text(
                            '${_missed.round()}',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              color: LoanlyTheme.accent,
                            ),
                          ),
                        ],
                      ),
                      SliderTheme(
                        data: SliderTheme.of(context).copyWith(
                          activeTrackColor: LoanlyTheme.accent,
                          inactiveTrackColor: LoanlyTheme.border,
                          thumbColor: LoanlyTheme.accent,
                        ),
                        child: Slider(
                          min: 0,
                          max: 12,
                          divisions: 12,
                          value: _missed.clamp(0, 12),
                          onChanged: (v) => setState(() => _missed = v),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _loading ? null : _submit,
                  icon: _loading
                      ? const SizedBox(
                          width: 22,
                          height: 22,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.black,
                          ),
                        )
                      : const Icon(Icons.radar_rounded),
                  label: Text(_loading ? 'Analyzing servicing signals…' : 'Run monitoring'),
                ),
              ),
              if (_error != null) ...[
                const SizedBox(height: 16),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: const Color(0xFF2A1515),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: const Color(0xFF5C2020)),
                  ),
                  child: Text(
                    _error!,
                    style: const TextStyle(color: Color(0xFFFFB4B4), height: 1.35),
                  ),
                ),
              ],
              if (_result != null) ...[
                const SizedBox(height: 24),
                _StatusBanner(result: _result!, chipColor: _chipColor),
                const SizedBox(height: 18),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Probability breakdown',
                          style: TextStyle(fontWeight: FontWeight.w700),
                        ),
                        const SizedBox(height: 14),
                        _ProbRow(label: 'Model PD', value: pct2(_result!.modelPd)),
                        _ProbRow(
                          label: 'Behavioural overlay',
                          value: pct2(_result!.behaviouralPd),
                        ),
                        const Divider(height: 28),
                        _ProbRow(
                          label: 'Combined PD',
                          value: pct2(_result!.combinedPd),
                          emphasize: true,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Risk score ${_result!.riskScore.toStringAsFixed(1)} · Principal repaid '
                          '${(_result!.principalRepaidRatio * 100).toStringAsFixed(0)}%',
                          style: const TextStyle(color: LoanlyTheme.textMuted, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 14),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Row(
                          children: [
                            Icon(Icons.lightbulb_outline_rounded, color: LoanlyTheme.accent),
                            SizedBox(width: 8),
                            Text(
                              'Recommended action',
                              style: TextStyle(fontWeight: FontWeight.w700),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Text(
                          _result!.recommendedAction,
                          style: const TextStyle(height: 1.45),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 14),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Servicing notes',
                          style: TextStyle(fontWeight: FontWeight.w700),
                        ),
                        const SizedBox(height: 12),
                        for (final n in _result!.notes)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  '• ',
                                  style: TextStyle(color: LoanlyTheme.accent),
                                ),
                                Expanded(child: Text(n, style: const TextStyle(height: 1.35))),
                              ],
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ]),
          ),
        ),
      ],
    );
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({
    required this.result,
    required this.chipColor,
  });

  final PostLoanResult result;
  final Color Function(String chip) chipColor;

  @override
  Widget build(BuildContext context) {
    final c = chipColor(result.statusChip);
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        color: c.withValues(alpha: 0.12),
        border: Border.all(color: c.withValues(alpha: 0.45)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.health_and_safety_rounded, color: c, size: 36),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.status.replaceAll('_', ' '),
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: c,
                      ),
                ),
                const SizedBox(height: 6),
                Text(
                  'Missed payments (rolling): ${result.missedPayments} · Age on books: '
                  '${result.monthsSinceDisbursement} mo',
                  style: const TextStyle(color: LoanlyTheme.textMuted, fontSize: 13),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ProbRow extends StatelessWidget {
  const _ProbRow({
    required this.label,
    required this.value,
    this.emphasize = false,
  });

  final String label;
  final String value;
  final bool emphasize;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              color: emphasize ? LoanlyTheme.textPrimary : LoanlyTheme.textMuted,
              fontWeight: emphasize ? FontWeight.w700 : FontWeight.w400,
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontWeight: emphasize ? FontWeight.w800 : FontWeight.w600,
              fontSize: emphasize ? 18 : 15,
              color: emphasize ? LoanlyTheme.accent : LoanlyTheme.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}
