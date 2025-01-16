@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

rem 运行 Python 脚本
.\python_embeded\python.exe -s SimpleSDXL\model_checker.py

rem 获取脚本所在目录的上一级目录
for %%I in ("%~dp0..") do set "root=%%~fI\"

echo Root directory: !root!
echo.
if "!root:~-2!"=="\\" (
    echo 错误：非法路径，主目录不能为磁盘根目录！请创建SimpleAI主目录。
    echo.
    pause
    cmd
)

for /F %%i in ('dir /B "!root!*.zip" ^| find /C /V ""') do set "file_count=%%i"

set "loop_count=0"
for %%f in ("!root!*.zip") do (
    set /a loop_count+=1
    if !loop_count! equ 1 (
        echo ...
        echo ...
        echo ...
        echo ...
        echo ...
        echo ----------------------------------------------------------
        echo 准备解压!root!目录下的!file_count!个zip模型包，时间较长，需要耐心等待。安装完成后，请及时移除zip包，不必多次安装。安装过程中若中断，会导致解压出的文件残缺，需重新启动本程序再次解压覆盖。有任何疑问可进SimpleSDXL的QQ群求助：938075852      
    )
    echo 开始解压模型包：%%f
    echo 按任意键继续。
    pause > nul
    powershell -nologo -noprofile -command "& {Expand-Archive -Path '%%f' -DestinationPath '!root!' -Force}"
)

echo All done.
echo 按任意键继续。
pause > nul
cmd