import os
import sys


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)


try:
    import pygit2
    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))

    branch_name = repo.head.shorthand
    
    remote_name = 'origin'
    remote = repo.remotes[remote_name]

    remote.fetch()

    origin_name = 'main'
    main_name = 'SimpleSDXL'
    dev_name = 'SimpleSDXL_dev'
    local_branch_ref = f'refs/heads/{branch_name}'
    if '--dev' in (sys.argv):
        if branch_name != dev_name:
            branch_name = dev_name
            checkout_flag = True
            print(f'Ready to checkout {branch_name}')
            local_branch_ref = f'refs/heads/{branch_name}'
            branch_ref = repo.lookup_branch(branch_name)
            print(f'branch_ref:{branch_ref}')
            repo.checkout(branch_ref)
    else:
        if branch_name != main_name:
            branch_name = main_name
            checkout_flag = True
            print(f'Ready to checkout {branch_name}')
            local_branch_ref = f'refs/heads/{branch_name}'
            branch_ref = repo.lookup_branch(branch_name)
            print(f'branch_ref:{branch_ref}')
            repo.checkout(branch_ref)

    local_branch = repo.lookup_reference(local_branch_ref)
    local_commit = repo.revparse_single(local_branch_ref)

    import enhanced.version as version
    version.branch = f'{branch_name}'
    version.commit_id = f'{local_commit.id}'[:7]

    remote_reference = f'refs/remotes/{remote_name}/{branch_name}'
    remote_commit = repo.revparse_single(remote_reference)

    merge_result, _ = repo.merge_analysis(remote_commit.id)

    if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
        print(f'{branch_name}: Already up-to-date, {str(local_commit.id)[:7]}')
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
        local_branch.set_target(remote_commit.id)
        repo.head.set_target(remote_commit.id)
        repo.checkout_tree(repo.get(remote_commit.id))
        repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
        version.commit_id = f'{remote_commit.id}'[:7]
        print(f'{branch_name}: Fast-forward merge, {str(local_commit.id)[:7]} <- {str(remote_commit.id)[:7]}')
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
        print(f'{branch_name}: Update failed - Did you modify any file? {str(local_commit.id)[:7]} <- {str(remote_commit.id)[:7]}')
except Exception as e:
    print(f'{branch_name}: Update failed. {str(local_commit.id)[:7]} <- {str(remote_commit.id)[:7]}')
    print(str(e))

print(f'{branch_name}: Update succeeded.')
from launch import *
