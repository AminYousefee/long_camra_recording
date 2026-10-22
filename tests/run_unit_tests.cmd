@echo off
title Amin Studio unit tests (no cameras needed)
cd /d "%~dp0\.."
where python >nul 2>nul && (python -m unittest discover -s tests -v) || (py -3 -m unittest discover -s tests -v)
timeout /t 120
