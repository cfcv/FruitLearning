import argparse
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-src', '--source', required=True, help='source directory')
args = vars(ap.parse_args())

files = os.listdir(args['source'])
newWidth = 100
newHeight = 100
M_height = 0
M_width = 0

for i,f in enumerate(files):
    image = cv2.imread(f, cv2.CV_LOAD_IMAGE_COLOR)
    height, width, depth = image.shape
    M_height += height
    M_width += width
    #newImage = cv2.resize(img, (int(newHeight), int(newWidth)))
    #cv2.imwrite(name, newImage)

print("mean height: ", M_height/len(f))
print("mean width: ", M_width/len(f))


