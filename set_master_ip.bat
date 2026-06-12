@echo off
REM ===== Make THIS PC the MASTER (static ethernet IP 192.168.1.100) =====
echo Setting Ethernet to static IP 192.168.1.100 ...
netsh interface ip set address name="Ethernet" static 192.168.1.100 255.255.255.0
echo.
echo Done. Current Ethernet config:
netsh interface ip show address name="Ethernet"
echo.
pause
