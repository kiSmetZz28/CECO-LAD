import ndjson
import numpy as np
import psutil
import os
import pandas as pd

# Function to read an ndjson file and convert it to a list of JSON objects
def read_ndjson(file_path):
    with open(file_path, 'r') as file:
        data = ndjson.load(file)
    return data

def process_value(value):
    # {"1742158137.4325614": 
    #  {"8303": 
    #   {"threads": [[8303, 23347.71, 13.12]], 
    #    "io_counters": [1145, 12403, 0, 10127179776, 29787359, 10119250675], 
    #    "cpu_times": [23347.71, 13.12, 0.0, 0.0, 0.0], 
    #    "name": "executor_runner", 
    #    "memory_percent": 0.7030130537760423, 
    #    "cpu_percent": 0.0
    #    }
    #  }
    # }
    # each time read one line ndjson, add all the process together
    # first loop - key is time
    total_cpu = 0
    total_mem = 0
    
    cpu_times = []
    name = ''
    io = []
    for t, pro_l in value.items():
        # second loop - key is process id
        if pro_l:
            # print(pro_l)
            for id, p in pro_l.items():
                total_cpu = total_cpu + p['CPU']
                total_mem = total_mem + p['Mem_use']
                if (name == '' or name == p['Name']):
                    name = p['Name']
                io = p['io_counters']
            return float(t), name, total_cpu, total_mem, io
        else:
            return None

def system_value(value):
    # first loop - key is time
    for t, sys in value.items():
        return float(t), sys['CPU_total'], sys['Memory'], sys['temperature']['cpu_thermal'][0][1]

def process_monitor_info(path):
    json_list = read_ndjson(path)

    run_info = []
    cpu_l = []
    memory_l = []
    cpu_times_final = []
    io_counter = []

    # calculate average monitor info
    for item in json_list:
        info = process_value(item)
        if info != None:
            run_info.append(info)
        
            time, name, cpu, memory, io = info
            # print(cpu_times)

            cpu_l.append(cpu)
            memory_l.append(memory)
            # cpu_times_final = cpu_times
            io_counter = io

    start_time = run_info[0][0]
    end_time = run_info[-1][0]

    print(f"process {name} average cpu (%) usage is {np.average(cpu_l)}")
    print(f"process {name} average cpu (%) usage compare to total is {np.average(cpu_l)/28}")
    print(f"process {name} average memory usage is {np.average(memory_l)}")
    # print(f"process {name} average memory usage in B is {np.average(mem_use_l)}")
    return start_time, end_time, np.average(cpu_l)/28, np.average(memory_l), io_counter

def system_monitor_info(start_time, end_time, path):
    json_list = read_ndjson(path)

    cpu_l = []
    memory_l = []
    temperature_l = []

    # calculate average monitor info, after the process starts and before the process end.
    for item in json_list:
        info = system_value(item)
        if info != None:
            time, cpu, memory, temperature = info

            if time >= start_time and time <= end_time:
                cpu_l.append(cpu)
                memory_l.append(memory)
                temperature_l.append(temperature)

    print(f"system average cpu (%) usage is {np.average(cpu_l)}")
    print(f"system average memory (%) usage is {np.average(memory_l)}")
    print(f"system average temperature(C) is {np.average(temperature_l)}")
    return np.average(cpu_l), np.average(memory_l), np.average(temperature_l)

if __name__ == "__main__":
    start_time, end_time, avg_cpu_proc, avg_mem_p_proc, io_counter = process_monitor_info("./em_at_consumption/deeplog/hdfs/process_dl.ndjson")
    avg_cpu_sys, avg_mem_sys, avg_temp_sys = system_monitor_info(start_time, end_time, './em_at_consumption/deeplog/hdfs/system_dl.ndjson')
    # print(start_time)
    # print(end_time)
    # print(avg_cpu_proc)
    # print(io_counter)
    # avg_cpu_sys, avg_mem_sys, avg_temp_sys = system_monitor_info(start_time, end_time, 'E:/UND/2025Spring/origin/3/system_os.ndjson')

