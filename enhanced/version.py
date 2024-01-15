import os

branch = ''
commit_id = ''
simplesdxl_ver = ''

def get_simplesdxl_ver():
    global simplesdxl_ver, commit_id
    if not simplesdxl_ver:
        simplesdxl_log = os.path.abspath(f'./simplesdxl_log.md')
        line = ''
        if os.path.exists(simplesdxl_log):
            with open(simplesdxl_log, "r", encoding="utf-8") as log_file:
                line = log_file.readline().strip()
                while line:
                    if line.startswith("# "):
                        break
                    line = log_file.readline().strip()
        else:
            line = '# 2023-12-20'
        date = line.split(' ')[1].split('-')
        simplesdxl_ver = f'v{date[0]}{date[1]}{date[2]}'
        if commit_id:
            simplesdxl_ver += f'.{commit_id}'
    return simplesdxl_ver

