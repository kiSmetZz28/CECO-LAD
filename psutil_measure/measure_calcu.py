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
    # each time read one line ndjson, add all the process together
    # first loop - key is time
    total_cpu = 0
    total_mem = 0
    total_memory = 0
    for t, pro_l in value.items():
        # second loop - key is process id
        for id, p in pro_l.items():
            total_cpu = total_cpu + p['CPU']
            total_mem = total_mem + p['Memory']
            total_memory = total_memory + p['Mem_use']
    return float(t), total_cpu, total_mem, total_memory

def system_value(value):
    # first loop - key is time
    for t, sys in value.items():
        return float(t), sys['CPU_total'], sys['Memory'], sys['temperature']['cpu_thermal'][0][1]

def process_monitor_info(path):
    json_list = read_ndjson(path)

    run_info = []
    cpu_l = []
    memory_l = []
    mem_use_l = []

    # calculate average monitor info
    for item in json_list:
        info = process_value(item)
        if info != None:
            run_info.append(info)
        
            time, cpu, memory, mem_use = info

            cpu_l.append(cpu)
            memory_l.append(memory)
            mem_use_l.append(mem_use)

    start_time = run_info[0][0]
    end_time = run_info[-1][0]

    print(f"process average cpu (%) usage is {np.average(cpu_l)}")
    print(f"process average cpu (%) usage compare to total is {np.average(cpu_l)/4}")
    print(f"process average memory (%) usage is {np.average(memory_l)}")
    print(f"process average memory usage in B is {np.average(mem_use_l)}")
    return start_time, end_time, np.average(cpu_l)/4, np.average(memory_l), np.average(mem_use_l)

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
    start_time, end_time, avg_cpu_proc, avg_mem_p_proc, avg_mem_b_proc = process_monitor_info('/home/qinxuan.shi/Desktop/process_parallel.ndjson')
    avg_cpu_sys, avg_mem_sys, avg_temp_sys = system_monitor_info(start_time, end_time, '/home/qinxuan.shi/Desktop/system_parallel.ndjson')

#     folder = 'monitor_results'
#     data_source_folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

#     model_types = ['executorch_os_A8W4', 'executorch_os_A8W8', 'executorch_os_W8only']
#     metric_columns = []
#     system_columns = []

#     for model in model_types:
#         metric_columns.extend([f"{model}_cpu", f"{model}_mem_p", f"{model}_mem_b"])
#         system_columns.extend([f"{model}_cpu", f"{model}_mem", f"{model}_temp"])

#     # Final DataFrame rows
#     all_rows = []
#     system_rows = []

#     for data_source in data_source_folders:
#         data_source_path = folder + "/" + data_source
#         iteration_folders = [f for f in os.listdir(data_source_path) if os.path.isdir(os.path.join(data_source_path, f))]

#         for iter in iteration_folders:
#             row = {"data_source": data_source}
#             system_row = {"data_source": data_source}
            
#             metrics = []
#             system_metrics = []

#             for model_type in model_types:
#                 process_path = data_source_path + "/" + iter + f"/process_{model_type}.ndjson"
#                 system_path = data_source_path + "/" + iter + f"/system_{model_type}.ndjson"
                
#                 if os.path.exists(process_path):
#                     start_time, end_time, avg_cpu_proc, avg_mem_p_proc, avg_mem_b_proc = process_monitor_info(process_path)
#                     avg_cpu_sys, avg_mem_sys, avg_temp_sys = system_monitor_info(start_time, end_time, system_path)

#                     metrics.extend([avg_cpu_proc, avg_mem_p_proc, avg_mem_b_proc])
#                     system_metrics.extend([avg_cpu_sys, avg_mem_sys, avg_temp_sys])
#                 else:
#                     metrics.extend([None, None, None])
#                     system_metrics.extend([None, None, None])
#             row.update(dict(zip(metric_columns, metrics)))
#             system_row.update(dict(zip(system_columns, system_metrics)))
#             all_rows.append(row)
#             system_rows.append(system_row)

# df = pd.DataFrame(all_rows, columns=["data_source"] + metric_columns)
# df_sys = pd.DataFrame(system_rows, columns=["data_source"] + system_columns)

# output_csv = folder + "/combined_process_data.csv"
# output_csv_sys = folder + "/combined_system_data.csv"
# df.to_csv(output_csv, index=False)
# df_sys.to_csv(output_csv_sys, index=False)