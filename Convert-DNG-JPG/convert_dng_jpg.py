import rawpy
import imageio
import os
import shutil
import sys

myDNG = []
for root, directories, filenames in os.walk(str(os.getcwd())):
    print("Total Count of Files: " + str(len(filenames)))
    dngCount = 0
    for filename in filenames:
        if filename.endswith(".dng"):
            myDNG.append(filename)
            dngCount += 1

print("Total Count of DNG Files: " + str(dngCount))
if dngCount == 0:
    print("No DNG files to convert, exiting...")
    exit()
else:
    for i in range(len(myDNG)):
        check = myDNG[i].replace('.dng', '.png')
        if os.path.isfile(check) is not True:
            try:
                print("Working on " + str(myDNG[i]))
                raw = rawpy.imread(str(myDNG[i]))
                rgb = raw.postprocess()
                newName = myDNG[i].replace('.dng', '.png')
                newFile = os.path.join(root, newName)
                imageio.imsave(newFile, rgb)
                print("Done.")

                folderCheck = str(os.getcwd() + "\\" + str(myDNG[i]).replace('.dng', ''))
                if os.path.isfile(folderCheck) is not True:
                    print("Making folder " + folderCheck)
                    os.mkdir(folderCheck)
                newName = myDNG[i].replace('.dng', '.png')
                newFile = os.path.join(root, newName)
                raw.close()
                print("Moving " + str(myDNG[i]) + " to " + str(folderCheck))
                shutil.move(myDNG[i], folderCheck)
                print("Moving " + str(newFile) + " to " + str(folderCheck))
                shutil.move(newFile, folderCheck)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
        else:
            folderCheck = str(os.getcwd() + "\\" + str(myDNG[i]).replace('.dng', ''))
            if os.path.isfile(check) is not True:
                print("Making folder " + folderCheck)
                os.mkdir(folderCheck)
            newName = filename.replace('.dng', '.png')
            newFile = os.path.join(root, newName)
            print("Moving " + str(myDNG[i]) + " to " + str(folderCheck))
            shutil.move(myDNG[i], folderCheck)
            print("Moving " + str(newFile) + " to " + str(folderCheck))
            shutil.move(newFile, folderCheck)
        inp = input("")


