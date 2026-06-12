@echo off
REM ===== Make THIS PC the SLAVE (static ethernet IP 192.168.1.101) =====
echo Setting Ethernet to static IP 192.168.1.101 ...
netsh interface ip set address name="Ethernet" static 192.168.1.101 255.255.255.0
echo.
echo Done. Current Ethernet config:
netsh interface ip show address name="Ethernet"
echo.
pause
