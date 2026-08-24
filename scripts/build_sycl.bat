@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\2026.0\oneapi-vars.bat" >nul 2>&1
where icx
icx --version
cmake -G Ninja -B build-sycl -S . ^
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DPRESTO_WITH_LLAMACPP=ON -DPRESTO_WITH_SYCL=ON ^
  -DPRESTO_WITH_VULKAN=OFF -DPRESTO_BUILD_TESTS=OFF
if errorlevel 1 exit /b 1
cmake --build build-sycl --target presto --parallel 6
