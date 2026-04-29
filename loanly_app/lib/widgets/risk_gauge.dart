import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme/loanly_theme.dart';

/// Animated semi-circular gauge for 0–100 risk score (matches dashboard spirit).
class RiskGauge extends StatefulWidget {
  const RiskGauge({
    super.key,
    required this.value,
    required this.subtitle,
  });

  final double value;
  final String subtitle;

  @override
  State<RiskGauge> createState() => _RiskGaugeState();
}

class _RiskGaugeState extends State<RiskGauge> with SingleTickerProviderStateMixin {
  late AnimationController _c;
  late Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _c = AnimationController(vsync: this, duration: const Duration(milliseconds: 1100))
      ..forward();
    _anim = CurvedAnimation(parent: _c, curve: Curves.easeOutCubic);
    _anim.addListener(() => setState(() {}));
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final t = _anim.value;
    final v = widget.value.clamp(0, 100) * t;

    return AspectRatio(
      aspectRatio: 1.15,
      child: LayoutBuilder(
        builder: (context, c) {
          final size = math.min(c.maxWidth, c.maxHeight * 1.2);
          return Stack(
            alignment: Alignment.center,
            children: [
              CustomPaint(
                size: Size(size, size),
                painter: _GaugePainter(progress: v / 100),
              ),
              Positioned(
                bottom: size * 0.06,
                child: Column(
                  children: [
                    Text(
                      v.toStringAsFixed(1),
                      style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                            fontWeight: FontWeight.w800,
                            color: LoanlyTheme.accent,
                          ),
                    ),
                    Text(
                      'Risk score',
                      style: Theme.of(context).textTheme.labelSmall?.copyWith(
                            color: LoanlyTheme.textMuted,
                          ),
                    ),
                    const SizedBox(height: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: LoanlyTheme.border),
                      ),
                      child: Text(
                        widget.subtitle,
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  _GaugePainter({required this.progress});

  final double progress;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.92);
    final radius = size.width * 0.42;

    final bgPaint = Paint()
      ..color = LoanlyTheme.border.withValues(alpha: 0.35)
      ..strokeWidth = 12
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final fgPaint = Paint()
      ..shader = SweepGradient(
        colors: [
          LoanlyTheme.accent.withValues(alpha: 0.25),
          LoanlyTheme.accent,
        ],
        startAngle: math.pi,
        endAngle: 2 * math.pi,
      ).createShader(Rect.fromCircle(center: center, radius: radius))
      ..strokeWidth = 12
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    const start = math.pi;
    const sweep = math.pi;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      start,
      sweep,
      false,
      bgPaint,
    );

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      start,
      sweep * progress.clamp(0.0, 1.0),
      false,
      fgPaint,
    );
  }

  @override
  bool shouldRepaint(covariant _GaugePainter oldDelegate) =>
      oldDelegate.progress != progress;
}
