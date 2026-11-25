@echo off
rem Start the embpy MCP server with a Python that has embpy installed (Windows).
rem
rem .claude-plugin/plugin.json names scripts/embpy-mcp-launcher without an extension:
rem Claude Code runs that shell script on macOS and Linux, and this file on Windows. Both
rem start the first of these that can run the server:
rem
rem   1. EMBPY_PYTHON, the python.exe of an environment where embpy[mcp,models] is
rem      installed. When it is set it is the only candidate: if it cannot run the
rem      server, the launcher says why and stops.
rem   2. The environment that uv installs embpy into with
rem        uv tool install --python 3.12 --editable "<this repository>[mcp,models]"
rem      (found through "uv tool dir").
rem   3. python.exe, then python3.exe, on PATH.
rem
rem embpy_mcp_check.py decides whether a candidate can run the server. When none can, the
rem launcher prints the reason and what to do on stderr and, when an MCP client started
rem it, also returns them as the error of the client's initialize request, so that
rem "claude mcp list" and /mcp show them.
rem
rem Arguments are passed on to the server: "embpy-mcp-launcher --version" prints the
rem version of embpy that the plugin would use.

setlocal EnableExtensions DisableDelayedExpansion
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
for %%I in ("%HERE%\..") do set "ROOT=%%~fI"
set "CHECK=%HERE%\embpy_mcp_check.py"
set "TRIED="
if not defined PYTHONUTF8 set "PYTHONUTF8=1"

rem 1. EMBPY_PYTHON
if not defined EMBPY_PYTHON goto :uv
set "PY=%EMBPY_PYTHON:"=%"
set "HOW=EMBPY_PYTHON"
call :usable "%PY%"
if not errorlevel 1 goto :start
goto :fail_explicit

rem 2. The uv tool environment
:uv
set "UV="
for %%E in (exe cmd bat) do if not defined UV for %%I in (uv.%%E) do set "UV=%%~$PATH:I"
if not defined UV (
    set "TRIED=%TRIED%; uv: not installed"
    goto :path
)
set "TOOLS="
for /f "usebackq delims=" %%D in (`call "%UV%" tool dir 2^>nul`) do set "TOOLS=%%D"
set "PY=%TOOLS%\embpy\Scripts\python.exe"
if not defined TOOLS goto :uv_none
if not exist "%PY%" goto :uv_none
set "HOW=uv tool environment"
call :usable "%PY%"
if not errorlevel 1 goto :start
set "TRIED=%TRIED%; %PY%: %REASON%"
goto :path
:uv_none
set "TRIED=%TRIED%; uv: no embpy tool environment"

rem 3. python.exe, then python3.exe, on PATH
:path
call :on_path python.exe
if defined FOUND goto :start
call :on_path python3.exe
if defined FOUND goto :start
goto :fail_none

:start
set "MSG=starting the embpy MCP server with %PY% (%HOW%)"
call :say
call "%PY%" -m embpy.mcp.server %*
exit /b %errorlevel%

rem on_path NAME: if NAME is on PATH and can run the server, set FOUND, PY and HOW.
:on_path
set "FOUND="
set "PY=%~$PATH:1"
if not defined PY exit /b 0
set "HOW=%~1 on PATH"
call :usable "%PY%"
if not errorlevel 1 (
    set "FOUND=1"
    exit /b 0
)
set "TRIED=%TRIED%; %PY%: %REASON%"
exit /b 0

rem usable PYTHON: errorlevel 0 if PYTHON can run the server; otherwise REASON says why.
:usable
set "REASON="
call "%~1" "%CHECK%" >nul 2>&1
if not errorlevel 1 exit /b 0
for /f "usebackq delims=" %%L in (`call "%~1" "%CHECK%" 2^>^&1`) do if not defined REASON set "REASON=%%L"
if not defined REASON set "REASON=it did not run"
set "REASON=%REASON:"='%"
exit /b 1

:say
setlocal EnableDelayedExpansion
>&2 echo embpy-mcp-launcher: !MSG!
endlocal
exit /b 0

rem The messages are put together with delayed expansion, so that paths and reasons
rem with special characters stay text.
:fail_explicit
set Q=^"
setlocal EnableDelayedExpansion
set "MSG=EMBPY_PYTHON is !PY!, but that Python cannot run the embpy MCP server: !REASON!. Install embpy there with: !Q!!PY!!Q! -m pip install !Q!embpy[mcp,models]!Q!, or point EMBPY_PYTHON to another Python."
goto :reply

:fail_none
set Q=^"
setlocal EnableDelayedExpansion
set "MSG=No Python found that can run the embpy MCP server (tried: !TRIED:~2!). Install embpy in a Python environment with: pip install !Q!embpy[mcp,models]!Q!, then set EMBPY_PYTHON to that environment's python.exe before starting Claude Code; or, with uv: uv tool install --python 3.12 --editable !Q!!ROOT![mcp,models]!Q!"

rem Report MSG and exit. When an MCP client started the launcher (timeout.exe refuses
rem redirected input), answer each of its requests with MSG as the error until it
rem closes stdin. (Claude Code may first send a server/discover request, and after an
rem error goes on with initialize; exiting right after a reply can lose it.)
:reply
>&2 echo embpy-mcp-launcher: !MSG!
"%SystemRoot%\System32\timeout.exe" /t 0 >nul 2>&1
if not errorlevel 1 exit /b 1
set "M=!MSG:\=\\!"
set "M=!M:"=\"!"
set /a ANSWERED=0
:next_request
if !ANSWERED! geq 5 exit /b 1
set "REQ="
set /p "REQ="
if not defined REQ exit /b 1
set "R=!REQ: =!"
set "R=!R:"=#!"
rem A notification has no id.
if "!R:#id#:=!"=="!R!" goto :next_request
rem The id is either second, after "jsonrpc", or last (Claude Code's initialize request).
if "!R:~0,10!"=="{#jsonrpc#" (
    set "R=!R:*#id#:=!"
) else (
    call :last_id
)
set "ID=0"
for /f "tokens=1 delims=,}" %%A in ("!R!") do set "ID=%%A"
set "ID=!ID:#="!"
echo {"jsonrpc":"2.0","id":!ID!,"error":{"code":-32000,"message":"!M!"}}
set /a ANSWERED+=1
goto :next_request

:last_id
set "NEXT=!R:*#id#:=!"
if "!NEXT!"=="!R!" exit /b 0
set "R=!NEXT!"
goto :last_id
