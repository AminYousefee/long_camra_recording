@echo off
cd /d "%~dp0\.."
if not exist hwtest_output mkdir hwtest_output
where python >nul 2>nul && (python tests\hw_thermal_modes.py > hwtest_output\modes_console.txt 2>&1) || (py -3 tests\hw_thermal_modes.py > hwtest_output\modes_console.txt 2>&1)
