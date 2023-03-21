import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import func_timeout
from utils.recorder import RandomAttackRecorder

class RandomImageAttacker:
    def __init__(self, model_path, n_img, attack_img_base, original_img, attack_pixels, total_timeout, limit=None):
        from tensorflow import keras

        self.n_img = n_img # 代表每個 iteration 要產生幾張隨機攻擊的圖
        self.model = keras.models.load_model(model_path)
        self.attack_img_base = attack_img_base
        self.original_img = original_img
        self.attack_pixels = attack_pixels
        self.total_timeout = total_timeout
        self.limit = limit # example 0.1, [x*0.9, x*1.1]

        self.recorder: RandomAttackRecorder = None
        self.iter = 0
        
    
    def attack_once(self):
        self.recorder.iter_start()
        
        if self.limit is None:
            attack_imgs = self._get_all_0_255_img_multi_pixels()
        else:
            attack_imgs = self._get_all_limit_img_multi_pixels()
                    
        original_atk_imgs = np.append(self.original_img, attack_imgs, axis=0)
        all_attack_result = self.model.predict(original_atk_imgs).argmax(axis=1)
        pred_label = all_attack_result[0]

        all_attack_result = all_attack_result[1:]
                                
        atk_label = None
        atk_img = None
        tmp = all_attack_result != pred_label
        tmp = all_attack_result[tmp==True]
        is_attack_success = len(tmp) > 0
        
        if is_attack_success:
            atk_label = int(tmp[0])
            atk_img = attack_imgs[atk_label]
        
        self.recorder.iter_end()
        return is_attack_success, atk_label, atk_img
    
    
    def attack_loop(self):
        self.recorder = RandomAttackRecorder()
        self.recorder.original_label = self.get_original_img_label()
        
        def run():
            while True:
                is_attack_success, atk_label, atk_img = self.attack_once()
                self.iter += 1
                
                if is_attack_success:
                    self.recorder.find_adversarial_input(atk_img, atk_label)
                    break

        self.recorder.start()
        try:            
            func_timeout.func_timeout(self.total_timeout, run)
        except func_timeout.exceptions.FunctionTimedOut:
            self.recorder.total_timeout()
                
        self.recorder.end()

    
    def _get_all_0_255_img_multi_pixels(self):
        n_pixel = len(self.attack_pixels)

        # 0~255隨機改動多個pixel，產生 n_img 張攻擊的圖
        attack_imgs = np.repeat(np.expand_dims(self.attack_img_base, 0), self.n_img, axis=0)
        attack_comb = np.random.randint(0, 256, size=(self.n_img, n_pixel))

        for img, rand_values in zip(attack_imgs, attack_comb):
            for attack_pixel, rand_val in zip(self.attack_pixels, rand_values):
                row, col = attack_pixel
                img[row, col, 0] = rand_val

        attack_imgs = attack_imgs.astype("float32") / 255

        # 回傳 攻擊過的圖片 n_img 張
        return attack_imgs
    
    
    def _get_all_limit_img_multi_pixels(self):
        n_pixel = len(self.attack_pixels)

        # 隨機改動多個pixel，產生 n_img 張攻擊的圖 limit
        attack_imgs = np.repeat(np.expand_dims(self.attack_img_base, 0), self.n_img, axis=0)
        
        for img in attack_imgs:
            for attack_pixel in self.attack_pixels:
                row, col = attack_pixel
                cur_value = img[row, col, 0]
                lb = max(0, int(cur_value * (1-self.limit)))
                ub = min(255, int(cur_value * (1+self.limit)))
                rand_val = np.random.randint(lb, ub)
                
                img[row, col, 0] = rand_val

        attack_imgs = attack_imgs.astype("float32") / 255

        # 回傳 攻擊過的圖片 n_img 張
        return attack_imgs
    

    def get_original_img_label(self):        
        original_label = self.model.predict(self.original_img).argmax(axis=1)[0]
        original_label = int(original_label)
        return original_label

