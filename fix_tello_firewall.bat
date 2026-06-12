@echo off
REM ===== Allow Tello video (UDP 11111) + Python through Windows Firewall =====
echo Adding firewall rule for Tello video UDP 11111...
netsh advfirewall firewall add rule name="Tello Video UDP 11111" dir=in action=allow protocol=UDP localport=11111

echo Adding firewall rule for python.exe...
netsh advfirewall firewall add rule name="Python (Tello)" dir=in action=allow program="C:\Python314\python.exe" enable=yes

echo.
echo Done. Rules added:
netsh advfirewall firewall show rule name="Tello Video UDP 11111"
echo.
pause
