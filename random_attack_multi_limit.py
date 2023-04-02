# disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from multiprocessing import Process
import time
from utils.random_attack_exp import run_multi_attack


# model_name = "mnist_sep_act_m6_9628"
model_name = "mnist_sep_act_m6_9653_noise"
model_path = f'./mnist_model/{model_name}.h5'

# TOP_N_SHAP = 1
N_IMG_EACH_ATTACK = 20000
TOTAL_TIMEOUT = 600
NUM_PROCESS = 25
# 

if __name__ == "__main__":

    from utils.random_attack_exp import random_random_1_4_8_16_32_limit, random_shap_1_4_8_16_32_limit
    
    inputs = random_random_1_4_8_16_32_limit(
        model_name, model_path, 400, N_IMG_EACH_ATTACK, TOTAL_TIMEOUT, limit=70)
    
    inputs.extend(random_shap_1_4_8_16_32_limit(
        model_name, model_path, 400, N_IMG_EACH_ATTACK, TOTAL_TIMEOUT, limit=70)
    )
    

    print("#"*40, f"number of inputs: {len(inputs)}", "#"*40)
    time.sleep(3)

    ########## 分派input給各個subprocesses ##########    
    all_subprocess_tasks = [[] for _ in range(NUM_PROCESS)]
    cursor = 0
    for task in inputs:    
        all_subprocess_tasks[cursor].append(task)    
        
        cursor+=1
        if cursor == NUM_PROCESS:
            cursor = 0

    running_processes = []
    for sub_tasks in all_subprocess_tasks:
        if len(sub_tasks) > 0:
            p = Process(target=run_multi_attack, args=(sub_tasks, N_IMG_EACH_ATTACK, ))        
            p.start()
            running_processes.append(p)
            time.sleep(1) # subprocess start 的間隔時間
        
    for p in running_processes:
        p.join()

    print('done')
