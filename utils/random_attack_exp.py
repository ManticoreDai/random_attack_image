import os
import json
import numpy as np

def get_save_dir_from_save_exp(save_exp):
    save_dir = os.path.join("exp", save_exp['model_name'], save_exp['exp_name'], save_exp['input_name'])
    return save_dir


def run_multi_attack(args, n_img_each_attack):
    from utils.attacker import RandomImageAttacker
    
    for one_input, save_exp in args:        
        print(save_exp)
        attacker = RandomImageAttacker(*one_input)
        attacker.attack_loop()
        
        
        if save_exp is not None:
            recorder = attacker.recorder
            recorder.input_name = save_exp['input_name'] 
                        
            save_dir = get_save_dir_from_save_exp(save_exp)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            with open(os.path.join(save_dir, "stats.json"), 'w') as f:
                out_stats = recorder.output_stats_dict()
                out_stats['meta']['n_img_each_attack'] = n_img_each_attack
                json.dump(out_stats, f, indent="\t")
            

            if recorder.attack_label is not None:                
                img_name = f"adv_{recorder.original_label}_to_{recorder.attack_label}.jpg"
                print("#"*20, f'attack success: {img_name}', "#"*20)
                recorder.save_adversarial_input_as_image(os.path.join(save_dir, img_name))


##### Generate Inputs #####
def random_shap_3_5_10_hard(model_name, model_path, first_n_img, N_IMG_EACH_ATTACK, TOTAL_TIMEOUT):
    from utils.dataset import get_mnist_data

    x_test, x_test_255 = get_mnist_data()

    ### SHAP and hard image index
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')
    is_hard_img = np.load('./exp_result/MNIST-選點+選值/is_hard_img.npy')

    inputs = []

    for ton_n_shap in [3, 5, 10]:
        for idx, is_hard in zip(range(first_n_img), is_hard_img):
            if not is_hard:
                # 該張圖不是困難的就跳過
                continue

            save_exp = {
                "model_name": model_name,
                "input_name": f"mnist_test_{idx}",
                "exp_name": f"random/shap_{ton_n_shap}"
            }

            save_dir = get_save_dir_from_save_exp(save_exp)
            if os.path.exists(save_dir):
                # 已經有紀錄的圖跳過
                continue

            base_img = x_test_255[idx]
            norm_img = x_test[[idx]]
            attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap, :2].tolist()
            
            one_input = (model_path, N_IMG_EACH_ATTACK, base_img.copy(), norm_img.copy(), attack_pixels, TOTAL_TIMEOUT)
            inputs.append((one_input, save_exp))

    return inputs


def random_shap_1_4_8_16_32(model_name, model_path, first_n_img, N_IMG_EACH_ATTACK, TOTAL_TIMEOUT):
    from utils.dataset import get_mnist_data

    x_test, x_test_255 = get_mnist_data()

    ### SHAP
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')

    inputs = []
    for ton_n_shap in [1,4,8,16,32]:
        for idx in range(first_n_img):
            save_exp = {
                "model_name": model_name,
                "input_name": f"mnist_test_{idx}",
                "exp_name": f"random/shap_{ton_n_shap}"
            }

            save_dir = get_save_dir_from_save_exp(save_exp)
            if os.path.exists(save_dir):
                # 已經有紀錄的圖跳過
                continue

            base_img = x_test_255[idx]
            norm_img = x_test[[idx]]
            attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap, :2].tolist()
            
            one_input = (model_path, N_IMG_EACH_ATTACK, base_img.copy(), norm_img.copy(), attack_pixels, TOTAL_TIMEOUT)
            inputs.append((one_input, save_exp))

    return inputs


def random_random_1_4_8_16_32(model_name, model_path, first_n_img, N_IMG_EACH_ATTACK, TOTAL_TIMEOUT):
    from utils.dataset import get_mnist_data
    from utils.gen_random_pixel_location import mnist_test_data_10000

    x_test, x_test_255 = get_mnist_data()

    # random pixels location with fixed seed
    rando_pixels = mnist_test_data_10000()

    inputs = []
    for ton_n in [1,4,8,16,32]:
        for idx in range(first_n_img):
            save_exp = {
                "model_name": model_name,
                "input_name": f"mnist_test_{idx}",
                "exp_name": f"random/random_{ton_n}"
            }

            save_dir = get_save_dir_from_save_exp(save_exp)
            if os.path.exists(save_dir):
                # 已經有紀錄的圖跳過
                continue

            base_img = x_test_255[idx]
            norm_img = x_test[[idx]]
            attack_pixels = rando_pixels[idx, :ton_n, :2].tolist()
            
            one_input = (model_path, N_IMG_EACH_ATTACK, base_img.copy(), norm_img.copy(), attack_pixels, TOTAL_TIMEOUT)
            inputs.append((one_input, save_exp))

    return inputs


### with limit 0.1
def random_random_1_4_8_16_32_limit_01(model_name, model_path, first_n_img, N_IMG_EACH_ATTACK, TOTAL_TIMEOUT):
    from utils.dataset import get_mnist_data
    from utils.gen_random_pixel_location import mnist_test_data_10000

    limit = 0.1
    x_test, x_test_255 = get_mnist_data()

    # random pixels location with fixed seed
    rando_pixels = mnist_test_data_10000()

    inputs = []
    for ton_n in [1,4,8,16,32]:
        for idx in range(first_n_img):
            save_exp = {
                "model_name": model_name,
                "input_name": f"mnist_test_{idx}",
                "exp_name": f"random_limit/random_{ton_n}"
            }

            save_dir = get_save_dir_from_save_exp(save_exp)
            if os.path.exists(save_dir):
                # 已經有紀錄的圖跳過
                continue

            base_img = x_test_255[idx]
            norm_img = x_test[[idx]]
            attack_pixels = rando_pixels[idx, :ton_n, :2].tolist()
            
            one_input = (model_path, N_IMG_EACH_ATTACK, base_img.copy(), norm_img.copy(), attack_pixels, TOTAL_TIMEOUT, limit)
            inputs.append((one_input, save_exp))

    return inputs

def random_shap_1_4_8_16_32_limit_01(model_name, model_path, first_n_img, N_IMG_EACH_ATTACK, TOTAL_TIMEOUT):
    from utils.dataset import get_mnist_data
    limit = 0.1
    x_test, x_test_255 = get_mnist_data()

    ### SHAP
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')

    inputs = []
    for ton_n_shap in [1,4,8,16,32]:
        for idx in range(first_n_img):
            save_exp = {
                "model_name": model_name,
                "input_name": f"mnist_test_{idx}",
                "exp_name": f"random_limit/shap_{ton_n_shap}"
            }

            save_dir = get_save_dir_from_save_exp(save_exp)
            if os.path.exists(save_dir):
                # 已經有紀錄的圖跳過
                continue

            base_img = x_test_255[idx]
            norm_img = x_test[[idx]]
            attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap, :2].tolist()
            
            one_input = (model_path, N_IMG_EACH_ATTACK, base_img.copy(), norm_img.copy(), attack_pixels, TOTAL_TIMEOUT, limit)
            inputs.append((one_input, save_exp))

    return inputs


### with limit range +- N
def random_random_1_4_8_16_32_limit(model_name, model_path, first_n_img, N_IMG_EACH_ATTACK, TOTAL_TIMEOUT, limit):
    from utils.dataset import get_mnist_data
    from utils.gen_random_pixel_location import mnist_test_data_10000

    x_test, x_test_255 = get_mnist_data()

    # random pixels location with fixed seed
    rando_pixels = mnist_test_data_10000()

    inputs = []
    for ton_n in [1,4,8,16,32]:
        for idx in range(first_n_img):
            save_exp = {
                "model_name": model_name,
                "input_name": f"mnist_test_{idx}",
                "exp_name": f"random/limit_{limit}_{N_IMG_EACH_ATTACK}/random_{ton_n}"
            }

            save_dir = get_save_dir_from_save_exp(save_exp)
            if os.path.exists(save_dir):
                # 已經有紀錄的圖跳過
                continue

            base_img = x_test_255[idx]
            norm_img = x_test[[idx]]
            attack_pixels = rando_pixels[idx, :ton_n, :2].tolist()
            
            one_input = (model_path, N_IMG_EACH_ATTACK, base_img.copy(), norm_img.copy(), attack_pixels, TOTAL_TIMEOUT, limit)
            inputs.append((one_input, save_exp))

    return inputs

def random_shap_1_4_8_16_32_limit(model_name, model_path, first_n_img, N_IMG_EACH_ATTACK, TOTAL_TIMEOUT, limit):
    from utils.dataset import get_mnist_data
    x_test, x_test_255 = get_mnist_data()

    ### SHAP
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')

    inputs = []
    for ton_n_shap in [1,4,8,16,32]:
        for idx in range(first_n_img):
            save_exp = {
                "model_name": model_name,
                "input_name": f"mnist_test_{idx}",
                "exp_name": f"random/limit_{limit}_{N_IMG_EACH_ATTACK}/shap_{ton_n_shap}"
            }

            save_dir = get_save_dir_from_save_exp(save_exp)
            if os.path.exists(save_dir):
                # 已經有紀錄的圖跳過
                continue

            base_img = x_test_255[idx]
            norm_img = x_test[[idx]]
            attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap, :2].tolist()
            
            one_input = (model_path, N_IMG_EACH_ATTACK, base_img.copy(), norm_img.copy(), attack_pixels, TOTAL_TIMEOUT, limit)
            inputs.append((one_input, save_exp))

    return inputs
