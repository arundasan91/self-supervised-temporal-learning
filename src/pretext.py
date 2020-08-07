import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tms_tstep_dict = {}
for tms in np.linspace(0, 1500, 16):
    tms_tstep_dict[tms] = int(tms*87/1500)

def task_scaling(input_array, scaling_factor):
    """ Return input array scaled by scaling_factor.
    """
    return(np.multiply(scaling_factor, input_array))

def task_gaussian_noise(input_array, noise_factor):
    """ Return input array with added gaussian noise with mean 0 and variance noise_factor.
    """
    return(np.random.normal(0, noise_factor, input_array.shape))

def task_none(input_array, dummyfactor):
    """ Return the input array without any processing (identity)
    """
    return(input_array)

def task_zero(input_array, dummyfactor):
    """ Return zero
    """
    return(0)

factors = {task_scaling: [0.25,0.5,1.25,1.5],
          task_gaussian_noise: [0.1, 0.25, 0.5, 0.75, 0.9],
          task_none: [0],
          task_zero: [0]}

tasks = [task_zero] + [task_none] + \
        [task_gaussian_noise for i in range(len(factors[task_gaussian_noise]))] + \
        [task_scaling for i in range(len(factors[task_scaling]))]
        
label_df = pd.DataFrame({"task": tasks,
                         "factor": factors[task_zero] + factors[task_none] + \
                         factors[task_gaussian_noise] + factors[task_scaling],
                         "label": list(range(len(tasks)))})

def broken_window_and_task(input_array, task=None, factor=None, winsize=100, window=0, verbose=False):
    input_array_copy = input_array.copy()
    
    total_windows = int((1500//winsize)*2)
    if window < (total_windows//2):
        aust = 0
        auend = 8
        tstepst = tms_tstep_dict[window*winsize]
        tstepend = tms_tstep_dict[(window+1)*winsize]
    else:
        aust = 8
        auend = 17
        tstepst = tms_tstep_dict[(window-total_windows//2)*winsize]
        tstepend = tms_tstep_dict[(window+1-total_windows//2)*winsize]
    
    if task:
        input_array_copy[aust:auend][:,tstepst:tstepend] = task(input_array_copy[aust:auend][:,tstepst:tstepend], factor)
    else:
        raise ValueError("Please define a task first")
    
    if task == task_none:
        ywin_label = -1
    else:
        ywin_label = window
        
    ytask_label = int(label_df[np.multiply(label_df["task"]==task, label_df["factor"]==factor)]["label"])
    
    if verbose:
        print("winsize: {}, window: {}, aust: {}, auend: {}, tstepst: {}, tstepend: {}, task: {}, factor: {}".format(winsize, window, aust, auend, tstepst, tstepend, task, factor))
    return(input_array_copy, ywin_label, ytask_label)

def apply_broken_window_and_task(input_au_array, winsize=None, subject=None, paradigm=None, label=None, savearray=False, returnarray=True, verbose=False):

    x = []
    ywin = []
    ytask = []
    total_windows = int((1500//winsize)*2)
    
    tasks = [task_scaling, task_gaussian_noise, task_none, task_zero]
    
    if len(input_au_array.shape)>2:

        for eacharray in input_au_array.copy():
            for window in range(total_windows):
                for task in tasks:
                    for factor in factors[task]:
                        _x, _ywin_label, _ytask_label = broken_window_and_task(eacharray, task, factor, winsize=winsize, window=window, verbose=verbose)
                        x.append(_x)
                        ywin.append(_ywin_label)
                        ytask.append(_ytask_label)
                
    else:
        for window in range(total_windows):
            for task in tasks:
                for factor in factors[task]:
                    _x, _ywin_label, _ytask_label = broken_window_and_task(eacharray, task, factor, winsize=winsize, window=window, verbose=verbose)
                    x.append(_x)
                    ywin.append(_ywin_label)
                    ytask.append(_ytask_label)

    x = np.stack(x, 0)
    ywin = np.stack(ywin)
    ytask = np.stack(ytask)
    
    # print(str(subject), str(paradigm), str(label), x.shape)
    if savearray:
        with open("/".join(['../dataset/broken_window/', str(subject), str(paradigm), str(label), "_".join(['bw', str(subject), str(paradigm), str(label), 'x.npy'])]),'wb') as f: np.save(f, x)
        with open("/".join(['../dataset/broken_window/', str(subject), str(paradigm), str(label), "_".join(['bw', str(subject), str(paradigm), str(label), 'ywin.npy'])]),'wb') as f: np.save(f, ywin)
        with open("/".join(['../dataset/broken_window/', str(subject), str(paradigm), str(label), "_".join(['bw', str(subject), str(paradigm), str(label), 'ytask.npy'])]),'wb') as f: np.save(f, ytask)
    
    if returnarray:
        return(x, ywin, ytask)
