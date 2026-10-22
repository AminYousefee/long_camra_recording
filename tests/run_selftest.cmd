@echo off
title Amin Studio hardware self-test (close Amin Studio first)
cd /d "%~dp0\.."
echo Running the hardware self-test: cameras, photos, ~80 s of recording...
echo (Amin Studio must be closed, or the cameras are busy.)
echo.
where python >nul 2>nul && (python tests\hw_selftest.py %*) || (py -3 tests\hw_selftest.py %*)
echo.
echo The report is in hwtest_output\selftest_*\report.txt
timeout /t 120
