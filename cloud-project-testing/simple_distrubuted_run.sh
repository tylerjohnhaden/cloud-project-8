# Run 1 server distributed (runs successfully)
python3 ../tools/launch.py -n 1 \
  --sync-dst-dir /home/ubuntu/cloud-project-8/cloud-project-testing \
  --launcher local \
  "python3 /home/ubuntu/cloud-project-8/cloud-project-testing/resnet_dist.py"

# Run 2 server distributed (hangs and does not run)
# python3 ../tools/launch.py -n 2 \
#   -H /home/ubuntu/cloud-project-8/cloud-project-testing/hosts \
#   --sync-dst-dir /home/ubuntu/cloud-project-8/cloud-project-testing \
#   --launcher ssh \
#   "python3 /home/ubuntu/cloud-project-8/cloud-project-testing/resnet_dist.py"


# Run 4 server distributed (hangs and does not run)
# python3 ../tools/launch.py -n 4 \
#   -H /home/ubuntu/cloud-project-8/cloud-project-testing/hosts \
#   --sync-dst-dir /home/ubuntu/cloud-project-8/cloud-project-testing \
#   --launcher ssh \
#   "python3 /home/ubuntu/cloud-project-8/cloud-project-testing/resnet_dist.py"
