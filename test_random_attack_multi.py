# disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from multiprocessing import Process
import numpy as np
import time
from utils.random_attack_exp import run_multi_attack


model_name = "mnist_sep_act_m6_9628"
model_path = f'./mnist_model/{model_name}.h5'

TOP_N_SHAP = 1
N_IMG_EACH_ATTACK = 256
TOTAL_TIMEOUT = 10
NUM_PROCESS = 2


if __name__ == "__main__":

    from utils.dataset import get_mnist_data
    x_test, x_test_255 = get_mnist_data()

    ### SHAP and hard image index
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')
    is_hard_img = np.load('./exp_result/MNIST-選點+選值/is_hard_img.npy')

    inputs = []
    for idx in [403, 443]:
        base_img = x_test_255[idx]
        norm_img = x_test[[idx]]
        attack_pixels = test_shap_pixel_sorted[idx, :TOP_N_SHAP, :2].tolist()
        
        save_exp = {
            "model_name": model_name,
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"random/shap_{TOP_N_SHAP}"
        }

        one_input = (model_path, N_IMG_EACH_ATTACK, base_img.copy(), norm_img.copy(), attack_pixels, TOTAL_TIMEOUT)
        inputs.append((one_input, save_exp))

    
    running_processes = []
    p = Process(target=run_multi_attack, args=(inputs[0:1], N_IMG_EACH_ATTACK))
    p.start()
    running_processes.append(p)

    p = Process(target=run_multi_attack, args=(inputs[1:2], N_IMG_EACH_ATTACK))
    running_processes.append(p)
    p.start()

    for p in running_processes:
        p.join()

    print('done')




