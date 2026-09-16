@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38 >nul 2>&1
set PKG_CONFIG_PATH=A:\ffmpeg-master-latest-win64-gpl-shared\lib\pkgconfig;A:\vcpkg\installed\x64-windows\lib\pkgconfig
%*
