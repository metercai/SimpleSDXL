import os
import sys

if len(sys.argv) < 2:
    print("Usage: python launch_with_commit.py <commit_hash>")
    sys.exit(1)
commit_hash = sys.argv[1]

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

from modules.launch_util import is_installed, run_pip

try:
    if not is_installed("simpleai_base"):
        run_pip(f"install simpleai_base -i https://pypi.org/simple", "simpleai_base")
    from simpleai_base import simpleai_base
    token = simpleai_base.init_local(f'SimpleSDXL_User')

    import pygit2
    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))

    branch_name = repo.head.shorthand

    commit = repo.get(commit_hash)

    repo.reset(commit.id, pygit2.GIT_RESET_HARD)
    repo.checkout_index()

except Exception as e:
    print(f'{branch_name} {commit.id}: checkout failed. ')
    print(str(e))

del sys.argv[1]
from launch import *
