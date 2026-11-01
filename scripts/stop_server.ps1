# Stops any leftover copy of the assistant server so a fresh start never fails
# with "address already in use" (a stale server also serves stale code).
$killed = 0
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*disaster_assistant.api.server*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $killed++ }
if ($killed -gt 0) { Write-Host "Stopped $killed old server process(es)."; Start-Sleep -Seconds 2 }
