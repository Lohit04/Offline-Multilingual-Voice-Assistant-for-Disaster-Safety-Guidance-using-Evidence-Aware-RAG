# Opens the ports the assistant server listens on, on ALL network profiles
# (Windows marks campus / phone-hotspot Wi-Fi as "Public", which blocks
# inbound connections by default -- the phone can't reach the laptop until
# this is done). Self-elevates: accept the UAC prompt.
#
# Ports: 8443 HTTPS UI, 8001 plain API (bundled app), 8080 APK download,
#        8000 plain-HTTP fallback UI.
# Undo with:  Remove-NetFirewallRule -DisplayName "Disaster Assistant*"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

Remove-NetFirewallRule -DisplayName "Disaster Assistant*" -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName "Disaster Assistant (TCP in)" -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort 8443,8001,8080,8000 -Profile Any | Out-Null
Write-Host "Firewall rule added for TCP 8443, 8001, 8080, 8000."
Start-Sleep -Seconds 3
