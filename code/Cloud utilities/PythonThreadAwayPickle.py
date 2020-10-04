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
from glob import glob
import copy
import QC_PriorModelGeneration2Pickle_final_cgp as work
import GravityInversionUtilities as GI
import LoadInputDataUtility as DI

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
                    folderoutput = 'Scratch'+str(self.gpu_id)+'/'
                    input['P']['folder']=folderoutput
                    input['P']['output_name'] = folderoutput+'noddy_out'
                    work.AnalyzeResults(input)
                    self.input_queue.task_done()
                    self.output_queue.put('complete- run #'+str(input['thread_num'])+' on cpu#'+str(self.gpu_id))
                except Exception:
#                    traceback.print_exc(file='exceptionfile.txt')
                    self.input_queue.task_done()
                    self.output_queue.put('There was an exception- run #'+str(input['thread_num'])+'on cpu#'+str(self.gpu_id))
        return

def GetGenericP():
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
    P['Grav_comboNormalizingFactor'] = 2.4
    P['Tracer_comboNormalizingFactor'] = 1.0
    P['FaultIntersection_comboNormalizingFactor'] = 2400
    P['GT_comboNormalizingFactor'] = 315
    P['Mag_comboNormalizingFactor'] = 330     
    
    P['iterationNum'] =0
        
    P['verbose']=True    
    P['SimulationShiftType']='Datum Shift'
    P['ErrorNorm']='L1'
    P['ErrorType']='Global'
    P['cubesize']=150
    P['xy_origin']=[316448, 4379166, 1200-4000]
    P['xy_extent'] = [8800, 9035,4000]
    
    DI.loadData(P)
    return P

if __name__ == "__main__":
   
    folder = 'HistoryFileTransferRandom/'
    
    historyfiles = glob(folder+'*.his')
    nFiles = len(historyfiles)
    print(nFiles)
    #sample the realizations
    num_threads =  nFiles
    tasks = []
    num_cpus = 96
    P = GetGenericP()
    for i in range(num_threads):
        print(i)
        params={}
        params['P'] = copy.deepcopy(P)
        params['fileNum'] =i
        params['hisfile'] = historyfiles[i]
        params['thread_num'] =i

        params['P']['fileNum'] =i
        params['P']['hisfile'] = historyfiles[i]
        params['P']['thread_num'] =i

        tasks.append(params)

#    HypParametersPD = pd.DataFrame(tasks)    
#    HypParametersPD.to_csv('parametersPkl.csv')
#    HypParametersPD.to_pickle('parametersPkl.pkl')
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