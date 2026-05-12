import runpy
import os
import sys

def main():
    root_dir = os.path.dirname(__file__)
    nested_dir = os.path.join(root_dir, 'Fire-Detector-Python-main')
    nested_script = os.path.join(nested_dir, 'fire.py')

    if not os.path.exists(nested_script):
        print(f'Nested script not found: {nested_script}')
        sys.exit(1)

    old_cwd = os.getcwd()
    try:
        os.chdir(nested_dir)
        runpy.run_path(nested_script, run_name='__main__')
    finally:
        os.chdir(old_cwd)

if __name__ == '__main__':
    main()
