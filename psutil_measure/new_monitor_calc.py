import ndjson
import numpy as np
import psutil

# Function to read an ndjson file and convert it to a list of JSON objects
def read_ndjson(file_path):
    with open(file_path, 'r') as file:
        data = ndjson.load(file)
    return data

def process_value(value):

    # {"1731684837.537655": {
    #     "7977": {
    #         "cpu_percent": 0.0, 
    #         "memory_full_info": [51867648, 131395584, 8859648, 42356736, 0, 83607552, 0, 49094656, 49305600, 0], 
    #         "memory_maps": [["/home/qinxuan/executorch/cmake-out/executor_runner", 5775360, 42364928, 5775360, 0, 0, 5767168, 8192, 5775360, 8192, 0], 
    #                         ["[anon]", 37306368, 76603392, 37306368, 0, 0, 0, 37306368, 37306368, 37306368, 0], 
    #                         ["[heap]", 5816320, 6828032, 5816320, 0, 0, 0, 5816320, 5816320, 5816320, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libc.so.6", 1187840, 1777664, 29696, 1167360, 0, 0, 20480, 1187840, 20480, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33", 1572864, 2633728, 260096, 1449984, 0, 65536, 57344, 1277952, 57344, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libgcc_s.so.1", 73728, 200704, 38912, 65536, 0, 0, 8192, 49152, 8192, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libm.so.6", 376832, 659456, 38912, 368640, 0, 0, 8192, 376832, 8192, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1", 180224, 180224, 17408, 163840, 0, 0, 16384, 180224, 16384, 0], 
    #                         ["[vvar]", 0, 8192, 0, 0, 0, 0, 0, 0, 0, 0], ["[vdso]", 4096, 4096, 0, 4096, 0, 0, 0, 4096, 0, 0], 
    #                         ["[stack]", 20480, 135168, 20480, 0, 0, 0, 20480, 20480, 20480, 0]], 
    #         "threads": [[7977, 530.66, 0.38]], 
    #         "name": "executor_runner", 
    #         "cpu_times": [530.65, 0.38, 0.0, 0.0, 0.0], 
    #         "memory_percent": 0.6235283342738207, 
    #         "io_counters": [41, 295, 262144, 230346752, 20315479, 230185169]}}}

    # {"1731692098.5584404": {
    #     "1723": {
    #         "io_counters": [28, 151, 262144, 115286016, 20208996, 115140334], 
    #         "memory_full_info": [51699712, 131395584, 8568832, 42356736, 0, 83607552, 0, 48873472, 49712128, 0], 
    #         "cpu_times": [530.77, 0.54, 0.0, 0.0, 0.0], 
    #         "memory_percent": 0.6319559843808439, 
    #         "name": "executor_runner", 
    #         "cpu_percent": 0.0, 
    #         "threads": [[1723, 530.77, 0.54]], 
    #         "memory_maps": [["/home/qinxuan/executorch/cmake-out/executor_runner", 5369856, 42364928, 5369856, 0, 0, 5361664, 8192, 5369856, 8192, 0], 
    #                         ["[anon]", 37306368, 76603392, 37306368, 0, 0, 0, 37306368, 37306368, 37306368, 0], 
    #                         ["[heap]", 5816320, 6828032, 5816320, 0, 0, 0, 5816320, 5816320, 5816320, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libc.so.6", 1183744, 1777664, 52224, 1163264, 0, 0, 20480, 1183744, 20480, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libgcc_s.so.1", 73728, 200704, 40960, 65536, 0, 0, 8192, 73728, 8192, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33", 1605632, 2633728, 894976, 1421312, 0, 126976, 57344, 1605632, 57344, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libm.so.6", 393216, 659456, 193536, 258048, 0, 126976, 8192, 393216, 8192, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1", 180224, 180224, 20480, 163840, 0, 0, 16384, 180224, 16384, 0], 
    #                         ["[vvar]", 0, 8192, 0, 0, 0, 0, 0, 0, 0, 0], ["[vdso]", 4096, 4096, 0, 4096, 0, 0, 0, 4096, 0, 0], 
    #                         ["[stack]", 16384, 135168, 16384, 0, 0, 0, 16384, 16384, 16384, 0]]}}}

    # {"1731697543.017816": {
    #     "1506": {
    #         "io_counters": [18, 43, 0, 28897280, 20127086, 28858653], 
    #         "name": "executor_runner", 
    #         "memory_full_info": [51699712, 131395584, 8568832, 42356736, 0, 83607552, 0, 48865280, 49454080, 0], 
    #         "cpu_times": [530.24, 0.49, 0.0, 0.0, 0.0], 
    #         "cpu_percent": 0.0, 
    #         "memory_maps": [["/home/qinxuan/executorch/cmake-out/executor_runner", 5369856, 42364928, 5369856, 0, 0, 5361664, 8192, 5369856, 8192, 0], 
    #                         ["[anon]", 37302272, 76603392, 37302272, 0, 0, 0, 37302272, 37302272, 37302272, 0], 
    #                         ["[heap]", 5812224, 6828032, 5812224, 0, 0, 0, 5812224, 5812224, 5812224, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libc.so.6", 1183744, 1777664, 51200, 1163264, 0, 0, 20480, 1183744, 20480, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33", 1605632, 2633728, 657408, 1421312, 0, 126976, 57344, 1605632, 57344, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libgcc_s.so.1", 73728, 200704, 29696, 65536, 0, 0, 8192, 73728, 8192, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/libm.so.6", 393216, 659456, 192512, 258048, 0, 126976, 8192, 393216, 8192, 0], 
    #                         ["/usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1", 180224, 180224, 20480, 163840, 0, 0, 16384, 180224, 16384, 0], 
    #                         ["[vvar]", 0, 8192, 0, 0, 0, 0, 0, 0, 0, 0], ["[vdso]", 4096, 4096, 0, 4096, 0, 0, 0, 4096, 0, 0], 
    #                         ["[stack]", 16384, 135168, 16384, 0, 0, 0, 16384, 16384, 16384, 0]], 
    #         "threads": [[1506, 530.26, 0.49]], 
    #         "memory_percent": 5.483058210251955}}}

    # first loop - key is time
    for t, pro_l in value.items():
        # second loop - key is process id
        for id, p in pro_l.items():
            if p['cpu_times'][0] != 0:
                return t, p['cpu_times'], p['io_counters']


def get_process_start_time():
    file_path = 'C:/Users/15147/Desktop/executorch/raspberry pi/new_monitor/r5/test_process_A8W4.ndjson'
    json_list = read_ndjson(file_path)

    # return io cunters read and write bytes
    for item in json_list:
        info = process_value(item)
        if info != None:
            time, cpu_time, io = info
            return time

def get_io():
    file_path = 'C:/Users/15147/Desktop/executorch/raspberry pi/new_monitor/r5/test_process_A8W4.ndjson'
    json_list = read_ndjson(file_path)

    # return io cunters read and write bytes
    for item in json_list:
        info = process_value(item)
        if info != None:
            time, cpu_time, io = info
            if cpu_time[0] >= 300:
                return time, io[2], io[3]


# def get_temperature():
#     file_path = 'C:/Users/15147/Desktop/executorch/raspberry pi/5/origin_system.ndjson'
#     json_list = read_ndjson(file_path)

#     cpu_l = []
#     memory_l = []

#     # calculate average monitor info, after the process starts and before the process end.
#     for item in json_list:
#         info = system_value(item)
#         if info != None:
#             time, cpu, memory = info
#             if time >= start_time and time <= end_time:
#                 cpu_l.append(cpu)
#                 memory_l.append(memory)

def system_value(value):
    # first loop - key is time
    # {"1731684351.3748715": {
    #     "cpu_times": {"user": 1159.13, "nice": 3.6, "system": 196.2, "idle": 4401.86, 
    #                   "iowait": 65.59, "irq": 0.0, "softirq": 0.7, "steal": 0.0, "guest": 0.0, "guest_nice": 0.0}, 
    #     "cpu_freq": {"current": 2400.0, "min": 1500.0, "max": 2400.0}, 
    #     "CPU_total": 25.2, 
    #     "Memory": {"total": 8318410752, "available": 6677106688, "percent": 19.7, "used": 1213988864, 
    #                "free": 4984446976, "active": 771940352, "inactive": 1869479936, "buffers": 28553216, 
    #                "cached": 2091421696, "shared": 226279424, "slab": 330956800}, 
    #     "swap_memory": {"total": 1073737728, "used": 17301504, "free": 1056436224, "percent": 1.6, "sin": 0, "sout": 0}, 
    #     "logi_cpu": [0.0, 0.0, 100.0, 0.0], 
    #     "disk_io": {"read_count": 59540, "write_count": 11527, "read_bytes": 2001591296, "write_bytes": 1396870656, 
    #                 "read_time": 107717, "write_time": 634563, "read_merged_count": 16616, "write_merged_count": 19650, "busy_time": 83372}, 
    #     "temperature": {"cpu_thermal": [["", 68.3, None, None]], "rp1_adc": [["", 60.12, None, None]]}}}

    # {"1731691726.422774": {
    #     "cpu_times": {"user": 282.63, "nice": 0.0, "system": 40.35, "idle": 2342.89, 
    #                   "iowait": 11.16, "irq": 0.0, "softirq": 0.07, "steal": 0.0, "guest": 0.0, "guest_nice": 0.0}, 
    #     "cpu_freq": {"current": 1800.0, "min": 600.0, "max": 1800.0}, 
    #     "CPU_total": 25.0, 
    #     "Memory": {"total": 8180903936, "available": 7663706112, "percent": 6.3, "used": 263647232, 
    #                "free": 7460659200, "active": 387510272, "inactive": 161013760, "buffers": 39866368, 
    #                "cached": 416731136, "shared": 3465216, "slab": 96911360}, 
    #     "swap_memory": {"total": 0, "used": 0, "free": 0, "percent": 0.0, "sin": 0, "sout": 0}, 
    #     "logi_cpu": [0.0, 0.0, 0.0, 100.0], 
    #     "disk_io": {"read_count": 9503, "write_count": 2210, "read_bytes": 368877568, "write_bytes": 65853440, 
    #                 "read_time": 35957, "write_time": 61752, "read_merged_count": 10532, "write_merged_count": 2471, "busy_time": 14252}, 
    #     "temperature": {"cpu_thermal": [["", 38.459, None, None]]}}}

    # {"1731697020.8123562": {
    #     "cpu_times": {"user": 281.4, "nice": 0.0, "system": 60.41, "idle": 1847.22, 
    #                   "iowait": 16.69, "irq": 0.0, "softirq": 1.2, "steal": 0.0, "guest": 0.0, "guest_nice": 0.0}, 
    #     "cpu_freq": {"current": 1400.0, "min": 600.0, "max": 1400.0}, 
    #     "CPU_total": 25.3, 
    #     "Memory": {"total": 942899200, "available": 622653440, "percent": 34.0, "used": 239427584, 
    #                 "free": 341782528, "active": 329359360, "inactive": 117616640, "buffers": 42668032, 
    #                 "cached": 319021056, "shared": 3317760, "slab": 93999104}, 
    #     "swap_memory": {"total": 0, "used": 0, "free": 0, "percent": 0.0, "sin": 0, "sout": 0}, 
    #     "logi_cpu": [0.0, 0.0, 100.0, 0.0], 
    #     "disk_io": {"read_count": 8744, "write_count": 1550, "read_bytes": 338124288, "write_bytes": 30275584, 
    #                 "read_time": 46175, "write_time": 95507, "read_merged_count": 12095, "write_merged_count": 1256, "busy_time": 21752}, 
    #     "temperature": {"cpu_thermal": [["", 43.47, None, None]]}}}

    for t, sys in value.items():
        return float(t), sys


def process_monitor_info():
    file_path = 'C:/Users/15147/Desktop/executorch/raspberry pi/5/origin_processes.ndjson'
    # file_path = 'processes_A8W8.ndjson'
    # file_path = 'processes_A8W4.ndjson'
    # file_path = 'processes_A16W8.ndjson'
    # file_path = 'processes_origin.ndjson'

    # file_path = 'hdfs_processes_A8W4.ndjson'
    json_list = read_ndjson(file_path)

    run_info = []
    cpu_l = []
    memory_l = []
    mem_use_l = []

    # calculate average monitor info
    for item in json_list:
        info = process_value(item)
        if info != None:
            run_info.append(info)
        
            time, name, cpu, memory, mem_use = info

            cpu_l.append(cpu)
            memory_l.append(memory)
            mem_use_l.append(mem_use)

    start_time = run_info[0][0]
    end_time = run_info[-1][0]

    print(f"process {name} average cpu (%) usage is {np.average(cpu_l)}")
    print(f"process {name} average cpu (%) usage compare to total is {np.average(cpu_l)/4}")
    print(f"process {name} average memory (%) usage is {np.average(memory_l)}")
    print(f"process {name} average memory usage in B is {np.average(mem_use_l)}")
    return start_time, end_time

def system_monitor_info(start_time, end_time):
    file_path = 'C:/Users/15147/Desktop/executorch/raspberry pi/5/origin_system.ndjson'
    # file_path = 'system_A8W8.ndjson'
    # file_path = 'system_A8W4.ndjson'
    # file_path = 'system_A16W8.ndjson'
    # file_path = 'system_origin.ndjson'

    # file_path = 'hdfs_system_A8W4.ndjson'
    json_list = read_ndjson(file_path)

    cpu_l = []
    memory_l = []

    # calculate average monitor info, after the process starts and before the process end.
    for item in json_list:
        info = system_value(item)
        if info != None:
            time, cpu, memory = info
            if time >= start_time and time <= end_time:
                cpu_l.append(cpu)
                memory_l.append(memory)

    print(f"system average cpu (%) usage is {np.average(cpu_l)}")
    print(f"system average memory (%) usage is {np.average(memory_l)}")

if __name__ == "__main__":
    # start_time, end_time = process_monitor_info()
    # system_monitor_info(start_time, end_time)
    # io = get_io()
    # print(io)
    start = get_process_start_time()
    end, read, write = get_io()

    print((read)/((float(end) - float(start))/60))