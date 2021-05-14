#!/bin/bash
wirelessCard='FillInWithActualNetworkAdapterHere'
#NEED TO GET THE NETWORK CARD FROM SERVER
echo 'Input CNN model to train. Use other script for NLP'
read model
#Turn the model into the correct form for the script
modelScript = "native:$model:32:299"
echo "The model to be trained is $model"

echo 'Feed in a file of hosts/workers'
#Here, include localhost plus any other workers to be used
read workerFile
#Get the number of workers in the host file
numberOfWorkers=$(wc -l $workerFile)

echo 'Number of GPUs to use?'

read numberOfGPUs

throughputFile="${model}_Throughput_${numberOfGPUs}GPU_${numberOfWorkers}Workers"
echo "For $numberOfWorkers worker with $numberofGPUs GPU(s)"


echo 'Running with 10 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 10000000 -u 10000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile


echo 'Running with 9 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 9000000 -u 9000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile


echo 'Running with 8 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 8000000 -u 8000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile

echo 'Running with 7 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 7000000 -u 7000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile

echo 'Running with 6 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 6000000 -u 6000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile

echo 'Running with 5 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 5000000 -u 5000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile

echo 'Running with 4 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 4000000 -u 4000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile

echo 'Running with 3 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 3000000 -u 3000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile

echo 'Running with 2 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 2000000 -u 2000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile

echo 'Running with 1 Gbps'

while read ipAddr; do
  ssh ubuntu@$ipAddr ./wondershaper -a $wirelessCard -d 1000000 -u 1000000
done <$workerFile

python3 incubator-mxnet/example/image-classification/benchmark.py --worker_file $workerFile --worker_count $numberOfWorkers --gpu_count $numberOfGPUs --networks modelScript >> $throughputFile
echo -e '\n'>> $throughputFile

./wondershaper -c -a $wirelessCard

echo "Tests complete. Data written to $throughputFile."