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
from glob import glob
import copy

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
   
    folder = 'Z:/TransferThesisDraft/HistoryFileTransfer/'

    historyfiles = glob(folder+'*.his')
    nFiles = len(historyfiles)

    P={}
    P['CalcGravity']=True
    P['CalcMagnetics']=True
    P['CalcGraniteTop']=True
    P['CalcTracer']=True
    P['CalcFaultIntersection']=True
    P['DataTypes'] = ['Grav', 'Mag', 'Tracer', 'GT', 'FaultIntersection']
    P['JoinType']='LINES'
    P['errCalcMethodFaultIntersection']='Distance'
    P['MaxFaultIntersectionError'] = 500
    
    #############################
    ## 0 define the model grid boundaries + some other input stuff
    #############################  
     
    P['iterationNum'] =0
    
    
    Norm = pd.read_pickle('NormalizingFactorSummary.pkl')
    
    P['verbose']=False    
    P['Grav_comboNormalizingFactor'] = Norm['Grav']
    P['Mag_comboNormalizingFactor'] = Norm['Mag']
    P['Tracer_comboNormalizingFactor'] = Norm['Tracer']
    P['GT_comboNormalizingFactor'] = Norm['GT']
    P['FaultIntersection_comboNormalizingFactor'] = Norm['FaultIntersection']    
    P['SimulationShiftType']='Datum Shift'
    P['ErrorNorm']='L1'
    P['ErrorType']='Global'
    P['cubesize']=150
    P['xy_origin']=[316448, 4379166, 1200-4000]
    P['xy_extent'] = [8800, 9035,4000]
    
    DI.loadData(P)


    #sample the realizations
    num_threads =  nFiles
    tasks = []
    num_cpus = 16
    
    for i in range(num_threads):
        params = copy.deepcopy(P)
        
        params['fileNum'] =i
        params['thread_num'] =i
        folderoutput = 'Scratch/'
        params['folder']=folderoutput
        params['output_name'] = P['folder']+'noddy_out'
    
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