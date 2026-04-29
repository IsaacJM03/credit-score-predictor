import 'package:flutter/material.dart';

import '../models/loan_decision_result.dart';
import '../services/loanly_api.dart';
import '../theme/loanly_theme.dart';
import '../util/format_pct.dart';
import '../widgets/applicant_fields_form.dart';
import '../widgets/risk_gauge.dart';

class LoanApprovalScreen extends StatefulWidget {
  const LoanApprovalScreen({super.key, required this.api});

  final LoanlyApiService api;

  @override
  State<LoanApprovalScreen> createState() => _LoanApprovalScreenState();
}

class _LoanApprovalScreenState extends State<LoanApprovalScreen> {
  late final ApplicantFieldsController _fields;

  LoanDecisionResult? _result;
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
      final data = _fields.toApplicantInput();
      final out = await widget.api.loanDecision(data);
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

  @override
  Widget build(BuildContext context) {
    return CustomScrollView(
      physics: const BouncingScrollPhysics(),
      slivers: [
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(20, 12, 20, 120),
          sliver: SliverList(
            delegate: SliverChildListDelegate([
              _HeroBand(),
              const SizedBox(height: 20),
              Card(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: ApplicantFieldsForm(controller: _fields),
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
                      : const Icon(Icons.bolt_rounded),
                  label: Text(_loading ? 'Running pipeline…' : 'Run LOANLY pipeline'),
                ),
              ),
              if (_error != null) ...[
                const SizedBox(height: 16),
                _ErrorCard(message: _error!),
              ],
              if (_result != null) ...[
                const SizedBox(height: 28),
                _VerdictHeader(result: _result!),
                const SizedBox(height: 20),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: RiskGauge(
                        key: ValueKey(_result!.probabilityOfDefault),
                        value: _result!.riskScore.clamp(0, 100),
                        subtitle: _result!.riskBand,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: _MetricsColumn(result: _result!),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                _PipelineRow(),
                const SizedBox(height: 14),
                _EnsembleCard(result: _result!),
                const SizedBox(height: 16),
                _ReasonsCard(reasons: _result!.reasons),
              ],
            ]),
          ),
        ),
      ],
    );
  }
}

class _HeroBand extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: LinearGradient(
          colors: [
            LoanlyTheme.surfaceElevated,
            LoanlyTheme.surface.withValues(alpha: 0.9),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        border: Border.all(color: LoanlyTheme.border),
        boxShadow: [
          BoxShadow(
            color: LoanlyTheme.accent.withValues(alpha: 0.08),
            blurRadius: 32,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: LoanlyTheme.accent.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(
                  Icons.currency_exchange_rounded,
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
                      'LOANLY',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.w800,
                            letterSpacing: 1.2,
                            color: LoanlyTheme.accent,
                          ),
                    ),
                    Text(
                      'Five-stage smart approval — same engine as the HF Space model suite.',
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
        ],
      ),
    );
  }
}

class _VerdictHeader extends StatelessWidget {
  const _VerdictHeader({required this.result});
  final LoanDecisionResult result;

  @override
  Widget build(BuildContext context) {
    final approve = result.stage5Decision == 'APPROVE';
    final c = approve ? LoanlyTheme.accent : const Color(0xFFFF4D4D);
    return AnimatedContainer(
      duration: const Duration(milliseconds: 420),
      curve: Curves.easeOutCubic,
      padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        color: c.withValues(alpha: 0.12),
        border: Border.all(color: c.withValues(alpha: 0.45)),
      ),
      child: Row(
        children: [
          Icon(
            approve ? Icons.verified_rounded : Icons.cancel_rounded,
            color: c,
            size: 40,
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.stage5Decision,
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: c,
                      ),
                ),
                Text(
                  'Repayment outlook: ${result.repaymentPrediction} · Fraud ${result.fraudFlagged ? "flagged" : "clear"}',
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

class _MetricsColumn extends StatelessWidget {
  const _MetricsColumn({required this.result});
  final LoanDecisionResult result;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _MiniStat(
          label: 'Probability of default',
          value: pct1(result.probabilityOfDefault),
          icon: Icons.percent_rounded,
        ),
        const SizedBox(height: 10),
        _MiniStat(
          label: 'Post-loan monitor (early warning)',
          value: result.postLoanDefaultProbability != null
              ? pct1(result.postLoanDefaultProbability!)
              : '—',
          icon: Icons.show_chart_rounded,
        ),
        const SizedBox(height: 10),
        _MiniStat(
          label: 'Borrower segment',
          value: result.segmentLabel ?? '—',
          icon: Icons.groups_2_rounded,
        ),
        const SizedBox(height: 10),
        _MiniStat(
          label: 'DTI · LTI',
          value:
              '${result.derivedDti.toStringAsFixed(2)} · ${result.derivedLti.toStringAsFixed(2)}',
          icon: Icons.balance_rounded,
        ),
      ],
    );
  }
}

class _MiniStat extends StatelessWidget {
  const _MiniStat({
    required this.label,
    required this.value,
    required this.icon,
  });

  final String label;
  final String value;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: LoanlyTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: LoanlyTheme.border),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 20, color: LoanlyTheme.accent.withValues(alpha: 0.85)),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: const TextStyle(
                    color: LoanlyTheme.textMuted,
                    fontSize: 11,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PipelineRow extends StatelessWidget {
  final _steps = const [
    ('1', 'Entry', Icons.person_outline),
    ('2', 'Prep', Icons.auto_fix_high_rounded),
    ('3', 'ML', Icons.hub_rounded),
    ('4', 'Risk', Icons.analytics_rounded),
    ('5', 'Decision', Icons.dashboard_customize_rounded),
  ];

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Decision pipeline',
          style: TextStyle(
            fontWeight: FontWeight.w700,
            color: LoanlyTheme.accent,
          ),
        ),
        const SizedBox(height: 12),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            children: [
              for (var i = 0; i < _steps.length; i++) ...[
                _StepChip(
                  index: _steps[i].$1,
                  label: _steps[i].$2,
                  icon: _steps[i].$3,
                ),
                if (i < _steps.length - 1)
                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 4),
                    child: Icon(
                      Icons.chevron_right_rounded,
                      color: LoanlyTheme.textMuted,
                      size: 18,
                    ),
                  ),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

class _StepChip extends StatelessWidget {
  const _StepChip({
    required this.index,
    required this.label,
    required this.icon,
  });

  final String index;
  final String label;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: LoanlyTheme.border),
        color: LoanlyTheme.surfaceElevated,
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            index,
            style: const TextStyle(
              color: LoanlyTheme.accent,
              fontWeight: FontWeight.w800,
              fontSize: 12,
            ),
          ),
          const SizedBox(width: 8),
          Icon(icon, size: 18, color: LoanlyTheme.textPrimary),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
        ],
      ),
    );
  }
}

class _EnsembleCard extends StatelessWidget {
  const _EnsembleCard({required this.result});
  final LoanDecisionResult result;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          initiallyExpanded: false,
          tilePadding: const EdgeInsets.fromLTRB(18, 8, 18, 8),
          childrenPadding: const EdgeInsets.fromLTRB(18, 0, 18, 16),
          leading: const Icon(Icons.layers_rounded, color: LoanlyTheme.accent),
          title: const Text(
            'Model transparency',
            style: TextStyle(fontWeight: FontWeight.w700),
          ),
          subtitle: const Text(
            'Upfront classifiers · post-loan monitors',
            style: TextStyle(fontSize: 12, color: LoanlyTheme.textMuted),
          ),
          children: [
            const Align(
              alignment: Alignment.centerLeft,
              child: Text(
                'Default probability by model',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 12,
                  color: LoanlyTheme.textMuted,
                ),
              ),
            ),
            const SizedBox(height: 8),
            ...result.classifierScores.entries.map(
              (e) => _ProbLine(label: _prettyModel(e.key), value: pct2(e.value)),
            ),
            if (result.postLoanScores.isNotEmpty) ...[
              const SizedBox(height: 14),
              const Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Post-loan monitors (early warning)',
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                    color: LoanlyTheme.textMuted,
                  ),
                ),
              ),
              const SizedBox(height: 8),
              ...result.postLoanScores.entries.map(
                (e) => _ProbLine(label: _prettyModel(e.key), value: pct2(e.value)),
              ),
            ],
          ],
        ),
      ),
    );
  }

  static String _prettyModel(String key) {
    switch (key) {
      case 'xgb_cls':
        return 'XGBoost';
      case 'lr_cls':
        return 'Logistic regression';
      case 'dnn_cls':
        return 'Deep neural net';
      case 'xgb_post':
        return 'Post-loan · XGBoost';
      case 'lgbm_post':
        return 'Post-loan · LightGBM';
      default:
        return key;
    }
  }
}

class _ProbLine extends StatelessWidget {
  const _ProbLine({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(child: Text(label, style: const TextStyle(fontSize: 13))),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13)),
        ],
      ),
    );
  }
}

class _ReasonsCard extends StatelessWidget {
  const _ReasonsCard({required this.reasons});
  final List<String> reasons;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.checklist_rounded, color: LoanlyTheme.accent),
                SizedBox(width: 8),
                Text(
                  'Decision rationale',
                  style: TextStyle(fontWeight: FontWeight.w700),
                ),
              ],
            ),
            const SizedBox(height: 12),
            for (final r in reasons)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('◆ ', style: TextStyle(color: LoanlyTheme.accent, fontSize: 10)),
                    Expanded(child: Text(r, style: const TextStyle(height: 1.35))),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message});
  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF2A1515),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF5C2020)),
      ),
      child: Text(
        message,
        style: const TextStyle(color: Color(0xFFFFB4B4), height: 1.35),
      ),
    );
  }
}
