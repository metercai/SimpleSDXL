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

def get_simplesdxl_short_ver():
    ver = get_simplesdxl_ver()
    return ver.split('.')[1]

def get_branch():
    global branch, commit_id
    if not branch:
        import pygit2
        pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)
        repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))
        branch = repo.head.shorthand
        if branch=="main":
            branch = "Fooocus"
        elif branch=="SimpleSDXL_dev":
            branch = "SimpAI_dev"
        commit_id = f'{repo.head.target}'[:7]
    return branch

