import time
import numpy as np
import cv2


class RandomAttackRecorder:
   def __init__(self):
       # iters
       self.iter_wall_time = []
       self.iter_cpu_time = []
      
       # total
       self.total_wall_time = None
       self.total_cpu_time = None
       self.total_iter = 0
      
       # meta
       self.input_name = None
       self.input_shape = None
       self.original_label = None
       self.attack_label = None
       self.adversarial_input = None
       self.is_finish = False # finish all iteration or generate an adversarial input
       self.is_timeout = False


   def iter_start(self):
       self._iter_start_wall_time = time.time()       
       self._iter_start_cpu_time = time.process_time()

   def iter_end(self):
       self.iter_wall_time.append(time.time() - self._iter_start_wall_time)
       self.iter_cpu_time.append(time.process_time() - self._iter_start_cpu_time)
       self.total_iter += 1


   def start(self):
       self._start_wall_time = time.time()
       self._start_cpu_time = time.process_time()


   def end(self):
       self.total_wall_time = time.time() - self._start_wall_time
       self.total_cpu_time = time.process_time() - self._start_cpu_time
       self.is_finish = True


   def total_timeout(self):
       self.is_timeout = True
      

   def find_adversarial_input(self, adv_input, attack_label):
       self.adversarial_input = adv_input
       self.attack_label = attack_label
      
  
   def save_adversarial_input_as_image(self, save_path):
       if self.adversarial_input is not None:
           img_0_255 = self.adversarial_input.copy()
           img_0_255 = (img_0_255*255).astype(int)
           cv2.imwrite(save_path, img_0_255)
      

   def output_stats_dict(self):
       res = {
           "meta": dict(),
           "total": dict(),
           "iters": dict(),
       }
       res['meta']['input_name'] = self.input_name
       res['meta']['original_label'] = self.original_label
       res['meta']['attack_label'] = self.attack_label       
       res['meta']['is_finish'] = self.is_finish
       res['meta']['is_timeout'] = self.is_timeout
      
      
       res['total']['total_wall_time'] = self.total_wall_time
       res['total']['total_cpu_time'] = self.total_cpu_time
       res['total']['total_iter'] = self.total_iter


       res['iters']['iter_wall_time'] = self.iter_wall_time
       res['iters']['iter_cpu_time'] = self.iter_cpu_time

       return res

