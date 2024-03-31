#!bin/sh

model_path=$1   # Path to the model for the match
activation=$2   # Activation function of the model for the match
model_type=$3   # Type of the model for the match
log_dir=$4      # Path to the directory for saving logs
port1=$5        # Port number for the network match of the first table
port2=$6        # Port number for the network match of the second table



team_tag=model
num_server=2
date=`date '+%m%d'`
board_size=1000
board_setting=wb5/dataset_for_vs_wb5.json
board_id_1=0
board_id_2=0

echo CONFIG
echo -n 'the number of servers : '
echo $num_server
echo -n 'model path : '
echo $model_path
echo -n 'board size : '
echo $board_size
echo -n 'board setting : '
echo $board_setting

time=`date '+%Y-%m-%d %H:%M:%S'`
echo -n $time 
echo -n ' -LOG- start collecting'

mkdir ${log_dir}
mkdir ${log_dir}/board_log
mkdir ${log_dir}/server_log

echo -n 'table1 servers'

echo -n 'board_id : '
echo -n ${board_id_1} 
echo -n ',port : '
echo ${port1}
board_id_1_zeros=`printf "%04d" ${board_id_1}`
bridge-server -p $port1 -b ${board_setting} -r ${board_id_1} -o ${log_dir}/board_log/table1_board_${board_id_1_zeros}.json >& ${log_dir}/server_log/server_${port1}.txt &

echo 'table1 clients'
for i in N S
do
    echo -n '$port : '
    echo -n ${port1}
    echo -n ', location : '
    echo -n $i
    echo -n ', team_tag : '
    echo ${team_tag}
    echo -n 'activation : '
    echo -n ${activation}
    echo -n ', model_type : '
    echo -n ${model_type}
    echo -n ', model_path : '
    echo ${model_path}
    (python -m wb5.model_client_script -p $port1 -l $i -t $team_tag -a $activation -mt $model_type -m $model_path >& ${log_dir}/server_log/client_${port1}_${i}.txt &)
done

echo 'table2 servers'


echo -n 'board_id : '
echo -n ${board_id_2} 
echo -n ',port : '
echo ${port2}
board_id_2_zeros=`printf "%04d" ${board_id_2}`
bridge-server -p $port2 -b ${board_setting} -r ${board_id_2} -o ${log_dir}/board_log/table2_board_${board_id_2_zeros}.json >& ${log_dir}/server_log/server_${port2}.txt &

echo -n 'table2 clients'
for i in E W 
do
    echo -n '$port : '
    echo -n ${port2}
    echo -n ', location : '
    echo -n $i
    echo -n ', team_tag : '
    echo ${team_tag}
    echo -n 'activation : '
    echo -n ${activation}
    echo -n ', model_type : '
    echo -n ${model_type}
    echo -n ', model_path : '
    echo ${model_path}
    (python -m wb5.model_client_script -p $port2 -l $i -t $team_tag -a $activation -mt $model_type -m $model_path >& ${log_dir}/server_log/client_${port2}_${i}.txt &)
done