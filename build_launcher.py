import os
import shutil
import shared

win32_root = os.path.dirname(os.path.dirname(__file__))
python_embeded_path = os.path.join(win32_root, 'python_embeded')

is_win32_standalone_build = os.path.exists(python_embeded_path) and os.path.isdir(python_embeded_path)

win32_cmd = '''
@echo off
.\python_embeded\python.exe -s SimpleSDXL\{cmds} %*
echo All done.
pause
'''


def build_launcher():
    if not is_win32_standalone_build:
        return

    branchs = {"SimpleSDXL": "entry_with_update.py", "SimpleSDXL_dev": "entry_with_update.py --dev", "SimpleSDXL_without_update": "launch.py", "SimpleSDXL_commit": "launch_with_commit.py 56e5200"}

    for (name, cmd) in branchs.items():
        win32_cmd_preset = win32_cmd.replace('{cmds}', f'{cmd}')
        bat_path = os.path.join(win32_root, f'run_{name}.bat')
        if not os.path.exists(bat_path) or name=='SimpleSDXL_commit':
            with open(bat_path, "w", encoding="utf-8") as f:
                f.write(win32_cmd_preset)
    return

def ready_checker():
    if not is_win32_standalone_build:
        pass #return
    
    for filename in os.listdir(shared.root):
        if 'checker' in filename and filename.endswith(".bat"):
            source_file = os.path.join(shared.root, filename)
            target_file = os.path.join(win32_root, filename)

            shutil.copy2(source_file, target_file)
