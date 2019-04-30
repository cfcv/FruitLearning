import argparse
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-src', '--source', required=True, help='source directory')
ap.add_argument('-c', '--count', required=True, help='stating counting')
args = vars(ap.parse_args())

files = os.listdir(args['source'])
print("number of files in the directory:",len(files))
start_count = int(args['count'])
print(start_count)

for i,f in enumerate(files):
    new_name = str(i+start_count) + "." + f[-3:]
    os.rename(args['source'] + "/" + f, args['source'] + "/" + new_name)

print("number of files in the directory after:", len(os.listdir(args['source'])))
