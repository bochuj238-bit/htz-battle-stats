@echo off
chcp 65001 >nul 2>&1
title 幻唐志战斗统计工具
cd /d "%~dp0"

set "PY="
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto havepython

python -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto havepython

goto nopython

:havepython
%PY% "%~dp0tools\launcher.py"
exit /b %errorlevel%

:nopython
echo.
echo  ==================================================
echo   [X] Python not found
echo  ==================================================
echo.
echo   1. Open  https://www.python.org/downloads/
echo   2. Install Python 3.11 or newer
echo   3. IMPORTANT: check "Add python.exe to PATH"
echo      during installation, then run this file again
echo.
echo  --------------------------------------------------
echo   没找到 Python。请：
echo   1. 打开 https://www.python.org/downloads/
echo   2. 下载安装 Python 3.11 或更高版本
echo   3. 安装时务必勾选 "Add python.exe to PATH"
echo      装完再双击一次本文件
echo  --------------------------------------------------
echo.
pause
