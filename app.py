import subprocess
import os
from ModelsTested.cnn_test import returnLicenseNB

# Run first script
subprocess.run(['python', '2. Detection and Segmentation.py'])

licenseplateNB = returnLicenseNB()
print(licenseplateNB)

# Specify the directory path
directory_path = 'Dataset/Segmented Images'

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    # Check if the path is a file (not a directory)
    if os.path.isfile(file_path):
        # Delete the file
        os.remove(file_path)


