# !/bin/bash

PATTERN=$1
echo "test_pattern="$PATTERN

CONFIG_FILE=/home/ajno5/work/2_pattern/dgemm/config/whisper_rv64gcv_config.json
echo "configuration file="$CONFIG_FILE

LOGFILE=./whisper_run_log.txt
echo "logfile="$LOGFILE

OPTION=" --semihosting" 
OPTION+=" --counters"
#OPTION+=" --logfile "$LOGFILE

if [[ "$2" == "gdb" ]]; then
    echo "Running with debugger."
    gdb-multiarch $1 --ex "target remote | whisper --configfile "$CONFIG_FILE $OPTION "--gdb "$PATTERN
else
    echo "Running normally with: $1"
    #whisper --configfile $CONFIG_FILE --logfile run_log.txt $1
    whisper --configfile $CONFIG_FILE $OPTION $PATTERN 2>&1 | tee $LOGFILE 
fi
