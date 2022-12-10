import os
import sys

# find DNG format image and convert them to jpg or png format to can openvc-python read them.
myDNG = []
for root, directories, filenames in os.walk(str(os.getcwd())):
    print(f"Total Count of Files: {str(len(filenames))}")
    dngCount = 0
    for filename in filenames:
        if filename.endswith(".dng"):
            myDNG.append(filename)
            dngCount += 1

print(f"Total Count of DNG Files: {str(dngCount)}")


