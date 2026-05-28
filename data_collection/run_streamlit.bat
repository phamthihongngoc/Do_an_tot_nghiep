@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay .venv. Dang tao moi truong ao...
  python -m venv .venv
  if errorlevel 1 goto :error
)

set "PYTHON=.venv\Scripts\python.exe"
set "MPLCONFIGDIR=%CD%\.tmp\matplotlib"

if not exist ".tmp\matplotlib" mkdir ".tmp\matplotlib"

"%PYTHON%" -m streamlit --version >nul 2>&1
if errorlevel 1 (
  echo Thieu thu vien. Dang cai requirements.txt...
  "%PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)

echo Dang chay giao dien Streamlit...
echo Mo trinh duyet tai: http://localhost:8501
"%PYTHON%" -m streamlit run streamlit_app.py --server.headless=false --server.port=8501 --browser.gatherUsageStats=false
goto :end

:error
echo.
echo Khong chay duoc ung dung. Kiem tra lai Python/Internet hoac noi dung loi phia tren.
pause

:end
endlocal
` 