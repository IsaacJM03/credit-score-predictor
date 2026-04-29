$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
    Write-Host 'Flutter is not on PATH. Install from https://docs.flutter.dev/get-started/install/windows then re-run.' -ForegroundColor Yellow
    exit 1
}

# Merge Android/iOS/Web/Desktop runner scaffolding into this folder without overwriting lib/
flutter create . --project-name loanly_app --org com.loanly.app

flutter pub get
flutter analyze

Write-Host 'Done. Run: flutter run' -ForegroundColor Green
