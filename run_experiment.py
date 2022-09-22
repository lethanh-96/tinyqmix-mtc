from joblib import Parallel, delayed
import os

cmds = [
    'python3 main.py --scenario=test --n_node=12 --n_test_step=400 --n_test_episode=1800 --algorithm=qmix_lstm',
    'python3 main.py --scenario=test --n_node=24 --n_test_step=400 --n_test_episode=1800 --algorithm=qmix_lstm',
    'python3 main.py --scenario=test --n_node=48 --n_test_step=400 --n_test_episode=1800 --algorithm=qmix_lstm',
    'python3 main.py --scenario=test --n_node=96 --n_test_step=400 --n_test_episode=1800 --algorithm=qmix_lstm',
    # 'python3 main.py --scenario=test --n_node=12 --n_test_step=2400 --n_test_episode=300 --algorithm=qmix_lstm',
    # 'python3 main.py --scenario=test --n_node=24 --n_test_step=2400 --n_test_episode=300 --algorithm=qmix_lstm',
    # 'python3 main.py --scenario=test --n_node=48 --n_test_step=2400 --n_test_episode=300 --algorithm=qmix_lstm',
    # 'python3 main.py --scenario=test --n_node=96 --n_test_step=2400 --n_test_episode=300 --algorithm=qmix_lstm',
    # 'python3 main.py --scenario=test --n_node=12 --n_test_step=72000 --n_test_episode=1  --algorithm=qmix_lstm',
    # 'python3 main.py --scenario=test --n_node=24 --n_test_step=72000 --n_test_episode=1  --algorithm=qmix_lstm',
    # 'python3 main.py --scenario=test --n_node=48 --n_test_step=72000 --n_test_episode=1  --algorithm=qmix_lstm',
    # 'python3 main.py --scenario=test --n_node=96 --n_test_step=72000 --n_test_episode=1  --algorithm=qmix_lstm',
]

def f(cmd):
    os.system(cmd + ' 1> /dev/null 2> /dev/null')

Parallel(n_jobs=os.cpu_count())(delayed(f)(cmd) for cmd in cmds)