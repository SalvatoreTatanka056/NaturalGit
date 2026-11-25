@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   🌿 NaturalGit - Installer per Windows (1-Click)
echo ============================================================
echo.

:: 1. Verifica Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Python non trovato nel PATH.
    echo     Se hai il file ngit.exe in dist\, puoi usarlo direttamente senza Python!
    goto standalone_install
)

echo [v] Python trovato nel sistema.
set INSTALL_DIR=%LOCALAPPDATA%\NaturalGit\bin
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

:: 2. Installa pacchetto Python
echo [*] Installazione pacchetto NaturalGit...
python -m pip install -e "%~dp0." --no-build-isolation --no-deps >nul 2>&1

:: 3. Copia eseguibili e launcher
copy /y "%~dp0ngit.bat" "%INSTALL_DIR%\ngit.bat" >nul
copy /y "%~dp0naturalgit.bat" "%INSTALL_DIR%\naturalgit.bat" >nul

if exist "%~dp0dist\ngit.exe" (
    copy /y "%~dp0dist\ngit.exe" "%INSTALL_DIR%\ngit.exe" >nul
    echo [v] Eseguibile standalone ngit.exe copiato in %INSTALL_DIR%
)

:: 4. Aggiungi INSTALL_DIR al PATH utente se non presente
powershell -NoProfile -Command ^
    "$path = [Environment]::GetEnvironmentVariable('PATH', 'User');" ^
    "if ($path -notlike '*%INSTALL_DIR%*') {" ^
    "    [Environment]::SetEnvironmentVariable('PATH', $path + ';%INSTALL_DIR%', 'User');" ^
    "    Write-Host '[v] %INSTALL_DIR% aggiunto al PATH utente.';" ^
    "} else {" ^
    "    Write-Host '[v] %INSTALL_DIR% gia presente nel PATH.';" ^
    "}"

echo.
echo ============================================================
echo  🎉 Installazione completata con successo!
echo ============================================================
echo.
echo  Puoi digitare subito in qualsiasi terminale:
echo    ngit "mostrami lo stato dei file"
echo    ngit --help
echo.
pause
exit /b 0

:standalone_install
set INSTALL_DIR=%LOCALAPPDATA%\NaturalGit\bin
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if exist "%~dp0dist\ngit.exe" (
    copy /y "%~dp0dist\ngit.exe" "%INSTALL_DIR%\ngit.exe" >nul
    powershell -NoProfile -Command ^
        "$path = [Environment]::GetEnvironmentVariable('PATH', 'User');" ^
        "if ($path -notlike '*%INSTALL_DIR%*') {" ^
        "    [Environment]::SetEnvironmentVariable('PATH', $path + ';%INSTALL_DIR%', 'User');" ^
        "}"
    echo [v] ngit.exe installato in %INSTALL_DIR% e aggiunto al PATH!
) else (
    echo [!] Eseguibile ngit.exe non trovato in dist\.
)
echo.
pause
exit /b 0
