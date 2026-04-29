import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// LOANLY brand: yellow/gold on deep black (matches HF Space README).
abstract final class LoanlyTheme {
  static const Color accent = Color(0xFFF5C518);
  static const Color accentDim = Color(0xFFB88900);
  static const Color bg = Color(0xFF0A0A0B);
  static const Color surface = Color(0xFF141416);
  static const Color surfaceElevated = Color(0xFF1C1C1F);
  static const Color border = Color(0xFF2A2A2E);
  static const Color textPrimary = Color(0xFFF4F4F5);
  static const Color textMuted = Color(0xFF9CA3AF);

  static ThemeData dark() {
    final base = ThemeData(
      brightness: Brightness.dark,
      useMaterial3: true,
      scaffoldBackgroundColor: bg,
      colorScheme: ColorScheme.dark(
        surface: surface,
        primary: accent,
        onPrimary: Colors.black,
        secondary: accentDim,
        outline: border,
      ),
    );
    return base.copyWith(
      textTheme: GoogleFonts.spaceGroteskTextTheme(base.textTheme).apply(
        bodyColor: textPrimary,
        displayColor: textPrimary,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: bg.withValues(alpha: 0.92),
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: GoogleFonts.spaceGrotesk(
          fontSize: 22,
          fontWeight: FontWeight.w700,
          color: textPrimary,
        ),
      ),
      cardTheme: CardThemeData(
        color: surfaceElevated,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: border),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: accent, width: 1.5),
        ),
        labelStyle: const TextStyle(color: textMuted),
        floatingLabelStyle: const TextStyle(color: accent),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: accent,
          foregroundColor: Colors.black,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 16),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          textStyle: GoogleFonts.spaceGrotesk(
            fontWeight: FontWeight.w700,
            fontSize: 16,
          ),
        ),
      ),
      dividerTheme: const DividerThemeData(color: border),
    );
  }
}
