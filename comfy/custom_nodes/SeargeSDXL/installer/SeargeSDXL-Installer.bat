@echo off

if not exist SeargeSDXL-Installer.py goto :notcorrect

if exist python_embeded goto :portabledir
if exist ..\..\..\..\python_embeded goto :extensiondir

:notcorrect
echo Installer scripts not correctly installed
echo Please copy the files SeargeSDXL-Installer.bat and SeargeSDXL-Installer.py into the ComfyUI_windows_portable folder

goto :end

:portabledir
echo Installer running correctly in ComfyUI_windows_portable
python_embeded\python.exe -m pip install opencv-python
python_embeded\python.exe SeargeSDXL-Installer.py from_batch
goto :end

:extensiondir
echo Installer running within SeargeSDXL extension folder, this may work but is not supported
..\..\..\..\python_embeded\python.exe -m pip install opencv-python
..\..\..\..\python_embeded\python.exe SeargeSDXL-Installer.py from_batch
goto :end

echo "How did you get here?"

:end
pause
