@echo off
title Thermal camera unit check - follow the instructions
cd /d "%~dp0\.."
where python >nul 2>nul && (python tests\hw_thermal_unit_check.py) || (py -3 tests\hw_thermal_unit_check.py)
