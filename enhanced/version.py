import os

def get_simplesdxl_ver():
    simplesdxl_log = os.path.abspath(f'./simplesdxl_log.md')
    line = ''
    if os.path.exists(simplesdxl_log):
        with open(simplesdxl_log, "r", encoding="utf-8") as log_file:
            line = log_file.readline().strip()
    else:
        line = '# 2023-12-20'
    date = line.split(' ')[1].split('-')
    return f'v{date[0]}{date[1]}{date[2]}'

simplesdxl_ver = get_simplesdxl_ver()
