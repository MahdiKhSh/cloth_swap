@echo off
setlocal
rem Run from the project directory; keep existing Python environment and project data intact.
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8

rem Garment Bible ships as editable source; regenerate the UI after patching it.
if exist "studio-ui\.rebuild-required" goto :build
if not exist "studio-ui\dist\index.html" goto :build
goto :run

:build
pushd "studio-ui"
if not exist "node_modules\.bin\vite.cmd" (
  echo Installing web UI dependencies...
  call npm ci --no-fund --no-audit
  if errorlevel 1 goto :build_error
)
echo Building web UI (Garment Bible)...
call npm run build
if errorlevel 1 goto :build_error
if exist ".rebuild-required" del /q ".rebuild-required"
popd

:run
if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found at .venv\Scripts\python.exe
  echo Reuse or create your original project venv, then install requirements.txt.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m cloth_swap.studio %*
exit /b %errorlevel%

:build_error
popd
echo Building the UI failed. Check Node.js and npm availability.
pause
exit /b 1
