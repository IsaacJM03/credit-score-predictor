import 'package:flutter_test/flutter_test.dart';
import 'package:loanly_app/main.dart';

void main() {
  testWidgets('LOANLY app loads', (WidgetTester tester) async {
    await tester.pumpWidget(const LoanlyApp());
    expect(find.textContaining('LOANLY'), findsWidgets);
  });
}
