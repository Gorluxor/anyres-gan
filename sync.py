import os
import glob
def search_files(start_location):
    with open("paths.txt", "w") as f:
        for files in glob.glob(os.path.join(start_location,'**',"*.tfevents.*"), recursive=True):
            # write folder location of the files
            f.write(os.path.dirname(files) + "\n")

if __name__ == '__main__':
    search_files(start_location="training-runs")