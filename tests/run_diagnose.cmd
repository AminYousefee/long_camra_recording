@echo off
rem Hardware diagnosis for Amin Studio -- results go to hwtest_output\
cd /d "%~dp0\.."
if not exist hwtest_output mkdir hwtest_output
where python >nul 2>nul && (python tests\hw_diagnose.py > hwtest_output\diag_console.txt 2>&1) || (py -3 tests\hw_diagnose.py > hwtest_output\diag_console.txt 2>&1)
echo finished > hwtest_output\diag_finished.flag
