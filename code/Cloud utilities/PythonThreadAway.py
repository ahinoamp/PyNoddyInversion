# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:39:34 2020

@author: ahinoamp
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:55:26 2020

@author: ahinoamp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:14:32 2020

@author: ahinoamp
"""
import pandas as pd
import random   
import multiprocessing
import numpy as np
import sys
import CombineFilesJustValuesPropRotateNewGCP as algo


class Predictor(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, gpu_id):
        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.gpu_id = gpu_id

    def initialize_logging(self, num):
        sys.stdout = open('OutputFiles/'+str(num) + ".out", "a")
        sys.stderr = open('OutputFiles/'+str(num) + "err.out", "a")

    def run(self):
        while True:
            input = self.input_queue.get()
            if input is None:
                self.input_queue.task_done()
                self.output_queue.put("Finished with processor %d" % self.gpu_id)
                break
            else:
                try:
#                    self.initialize_logging(input['thread_num'])
                    algo.MakeFiles(input)
                    self.input_queue.task_done()
                    self.output_queue.put('complete- run #'+str(input['thread_num'])+' on cpu#'+str(self.gpu_id))
                except Exception:
#                    traceback.print_exc(file='exceptionfile.txt')
                    self.input_queue.task_done()
                    self.output_queue.put('There was an exception- run #'+str(input['thread_num'])+'on cpu#'+str(self.gpu_id))
        return

if __name__ == "__main__":
      
    #sample the realizations
    num_threads =  20
    tasks = []
    num_cpus = 4
    
    for i in range(num_threads):
        params = {}
        
        params['fileNum'] =i
        params['thread_num'] =i
            
        tasks.append(params)

    HypParametersPD = pd.DataFrame(tasks)    
    HypParametersPD.to_csv('parameters.csv')
    HypParametersPD.to_pickle('parameters.pkl')
    
    p_list = []
    input_queue = multiprocessing.JoinableQueue()
    output_queue = multiprocessing.Queue()
    for i in range(num_cpus):
        p = Predictor(input_queue, output_queue, i)
        p_list.append(p)

    for task in tasks:
        input_queue.put((task))

    for p in p_list:
        p.start()

    for i in range(num_cpus):
        input_queue.put(None)

    for i in range(len(tasks)+num_cpus):
        print(output_queue.get())

    input_queue.join()

    for p in p_list:
        p.join()