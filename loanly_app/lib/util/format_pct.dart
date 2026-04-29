/// Percent helpers — avoids brittle NumberFormat variants across `intl` versions.
String pct1(double v) => '${(v * 100).toStringAsFixed(1)}%';
String pct2(double v) => '${(v * 100).toStringAsFixed(2)}%';
