import numpy as np
import cv2
import TEMimagemanipulation
import argparse



# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

# root = Tkinter.Tk()
# root.withdraw()
# file_path = tkFileDialog.askopenfilename()
# root.destroy()

data = np.genfromtxt("lattice.dat")
# data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
pointsArray = TEMimagemanipulation.getPoints(data)
xpoints = pointsArray[0]
ypoints = pointsArray[1]


