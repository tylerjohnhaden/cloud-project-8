#!/bin/bash

wirelessCard='FillInWithActualNetworkAdapterHere'
echo 'Input CNN model to train. Use other script for NLP'
read model

echo 'The model to be trained is $model'

echo 'Feed in a file of hosts/workers'
#Here, include localhost plus any other workers to be used
read workerFile
#Get the number of workers in the host file
numberOfWorkers=$(wc -l $workerFile)

echo 'Number of GPUs to use?'

read numberOfGPUs

throughputFile="${model}_Throughput_${numberOfGPUs}GPU_${numberOfWorkers}Workers"
echo 'For $numberOfWorkers worker with $numberofGPUs GPU(s)'
echo 'Running with 10 Gbps'
./wondershaper -a $wirelessCard -d 10000000 -u 10000000

echo 'Running with 9 Gbps'



echo 'Running with 8 Gbps'

while read ipAddr; do
  ssh -i /path/my-key-pair.pem my-instance-user-name@$ipAddr ./wondershaper -a $wirelessCard -d 8000000 -u 8000000
done <$workerFile

#SSH in to each other machine in workerFile and rate limit
#Unsure what the hell to put here, however to run the benchmark then need to play with model name. Alternatively, hard code for 3 models
python benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count 1 --networks 'native:inception-v3:32:299' > $throughputFile
echo -e '\n'> $throughputFile

echo 'Running with 7 Gbps'
./wondershaper -a $wirelessCard -d 7000000 -u 7000000

echo 'Running with 6 Gbps'
./wondershaper -a $wirelessCard -d 6000000 -u 6000000

echo 'Running with 5 Gbps'
./wondershaper -a $wirelessCard -d 5000000 -u 5000000

echo 'Running with 4 Gbps'
./wondershaper -a $wirelessCard -d 4000000 -u 4000000

echo 'Running with 3 Gbps'
./wondershaper -a $wirelessCard -d 3000000 -u 3000000

echo 'Running with 2 Gbps'
./wondershaper -a $wirelessCard -d 2000000 -u 2000000

echo 'Running with 1 Gbps'
./wondershaper -a $wirelessCard -d 1000000 -u 1000000
