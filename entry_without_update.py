import os
import sys


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')

from launch import *
