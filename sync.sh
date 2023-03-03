#!/bin/bash

# Open the paths.txt file for reading
while IFS= read -r path; do
  # Navigate to the path
  echo "$path"
  # Run the wandb sync command
  wandb sync $path --sync-tensorboard --id $(basename $path)
done < paths.txt