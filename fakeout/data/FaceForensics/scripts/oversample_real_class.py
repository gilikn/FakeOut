import os
import subprocess


def duplicate_real_ff(directory_path, duplicates_num=3):
    for x in os.listdir(directory_path):
        if '.mp4' in x:
            for i in range(duplicates_num):
                subprocess.call(f"cp {x} {x.split('.')[0]}#dup{i+1}.mp4", shell=True, cwd=directory_path)
    return
