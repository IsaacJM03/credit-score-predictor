import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/loan_approval_screen.dart';
import 'screens/post_loan_screen.dart';
import 'services/loanly_api.dart';
import 'theme/loanly_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarBrightness: Brightness.dark,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: LoanlyTheme.bg,
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const LoanlyApp());
}

class LoanlyApp extends StatefulWidget {
  const LoanlyApp({super.key});

  @override
  State<LoanlyApp> createState() => _LoanlyAppState();
}

class _LoanlyAppState extends State<LoanlyApp> {
  String _baseUrl = kDefaultLoanlyBaseUrl;
  int _tab = 0;

  LoanlyApiService get _api => LoanlyApiService(baseUrl: _baseUrl);

  Future<void> _editEndpoint() async {
    final ctrl = TextEditingController(text: _baseUrl);
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: LoanlyTheme.surfaceElevated,
        title: const Text('API base URL'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'Your Hugging Face Space HTTPS URL (no trailing slash). '
                'Example: https://isaacmwesigwa-credit-score-predictor-models.hf.space',
                style: TextStyle(fontSize: 13, color: LoanlyTheme.textMuted),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: ctrl,
                decoration: const InputDecoration(
                  labelText: 'Base URL',
                  hintText: 'https://….hf.space',
                ),
                keyboardType: TextInputType.url,
              ),
            ],
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          ElevatedButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Save')),
        ],
      ),
    );
    if (ok == true && mounted) {
      setState(() => _baseUrl = ctrl.text.trim());
    }
    ctrl.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LOANLY',
      debugShowCheckedModeBanner: false,
      theme: LoanlyTheme.dark(),
      home: Scaffold(
        extendBodyBehindAppBar: true,
        appBar: AppBar(
          title: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('LOANLY'),
              Text(
                'Smart lending intelligence',
                style: Theme.of(context).textTheme.labelSmall?.copyWith(
                      color: LoanlyTheme.textMuted,
                      letterSpacing: 0.3,
                    ),
              ),
            ],
          ),
          actions: [
            IconButton(
              tooltip: 'API endpoint',
              onPressed: _editEndpoint,
              icon: const Icon(Icons.link_rounded),
            ),
          ],
        ),
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [
                LoanlyTheme.bg,
                Color(0xFF121015),
              ],
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
            ),
          ),
          child: SafeArea(
            child: IndexedStack(
              index: _tab,
              children: [
                LoanApprovalScreen(api: _api),
                PostLoanScreen(api: _api),
              ],
            ),
          ),
        ),
        bottomNavigationBar: NavigationBarTheme(
          data: NavigationBarThemeData(
            indicatorColor: LoanlyTheme.accent.withValues(alpha: 0.22),
            labelTextStyle: WidgetStateProperty.resolveWith((states) {
              if (states.contains(WidgetState.selected)) {
                return const TextStyle(
                  fontWeight: FontWeight.w700,
                  color: LoanlyTheme.accent,
                  fontSize: 12,
                );
              }
              return const TextStyle(color: LoanlyTheme.textMuted, fontSize: 12);
            }),
          ),
          child: NavigationBar(
            height: 68,
            selectedIndex: _tab,
            backgroundColor: LoanlyTheme.surface.withValues(alpha: 0.94),
            surfaceTintColor: Colors.transparent,
            labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
            onDestinationSelected: (i) => setState(() => _tab = i),
            destinations: const [
              NavigationDestination(
                icon: Icon(Icons.task_alt_outlined),
                selectedIcon: Icon(Icons.task_alt_rounded),
                label: 'Approval',
              ),
              NavigationDestination(
                icon: Icon(Icons.monitor_heart_outlined),
                selectedIcon: Icon(Icons.monitor_heart_rounded),
                label: 'Monitoring',
              ),
            ],
          ),
        ),
      ),
    );
  }
}
