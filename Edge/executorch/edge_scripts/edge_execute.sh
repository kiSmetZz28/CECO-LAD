#!/bin/bash
set -euo pipefail

LOG="run_qbat_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "Logging to: $LOG"

# --- Run bgl PARALLEL with monitoring ---

echo "Starting psutil monitoring for bgl PARALLEL tasks..."
python3 psutil_measure/monitor.py --file_name parallel  --data_source bgl &

monitor_par_pid=$!

echo "Waiting for 10 seconds..."
sleep 10

echo "Starting parallel execution (parallel.sh)..."
start_exec=$(date +%s%3N)
bash ./scripts/parallel_execute_bgl.sh
end_exec=$(date +%s%3N)
echo "parallel_execute_bgl.sh elapsed: $((end_exec - start_exec)) ms"

echo "Parallel execution finished. Waiting 10 seconds..."
sleep 20

echo "Stopping parallel monitor..."
kill $monitor_par_pid
if ps -p $monitor_par_pid > /dev/null; then
    echo "Force killing parallel monitor..."
    kill -9 $monitor_par_pid
fi

echo "All bgl tasks complete."


# --- Run PARALLEL with monitoring ---

echo "Starting psutil monitoring for os PARALLEL tasks..."
python3 psutil_measure/monitor.py --file_name parallel  --data_source os &

monitor_par_pid=$!

echo "Waiting for 10 seconds..."
sleep 10

echo "Starting parallel execution (parallel.sh)..."
start_exec=$(date +%s%3N)
bash ./scripts/parallel_execute_os.sh
end_exec=$(date +%s%3N)
echo "parallel_execute.sh elapsed: $((end_exec - start_exec)) ms"

echo "Parallel execution finished. Waiting 10 seconds..."
sleep 20

echo "Stopping parallel monitor..."
kill $monitor_par_pid
if ps -p $monitor_par_pid > /dev/null; then
    echo "Force killing parallel monitor..."
    kill -9 $monitor_par_pid
fi

echo "All os tasks complete."


# --- Run hdfs PARALLEL with monitoring ---

echo "Starting psutil monitoring for hdfs PARALLEL tasks..."
python3 psutil_measure/monitor.py --file_name parallel  --data_source hdfs &

monitor_par_pid=$!

echo "Waiting for 10 seconds..."
sleep 10

echo "Starting parallel execution (parallel.sh)..."
start_exec=$(date +%s%3N)
bash ./scripts/parallel_execute_hdfs.sh
end_exec=$(date +%s%3N)
echo "parallel_execute_hdfs.sh elapsed: $((end_exec - start_exec)) ms"

echo "Parallel execution finished. Waiting 10 seconds..."
sleep 20

echo "Stopping parallel monitor..."
kill $monitor_par_pid
if ps -p $monitor_par_pid > /dev/null; then
    echo "Force killing parallel monitor..."
    kill -9 $monitor_par_pid
fi

echo "All hdfs tasks complete."


