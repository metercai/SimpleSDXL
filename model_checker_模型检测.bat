@echo off
chcp 65001 > nul

.\python_embeded\python.exe -s SimpleSDXL\model_checker.py

echo All done.
echo 按任意键继续。
pause > nul
cmd