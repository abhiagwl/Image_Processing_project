from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Tkinter, tkFileDialog
from mblur2 import mblur
from redI3 import redI

class PointsBuilder:
  def __init__(self, data, imshow = True):
    self.imshow = imshow

    self.data = data
    self.fig = plt.figure()
    self.fig.suptitle("Select points by using the left mouse button.\nRemove the previous point by using the right mouse button.")
    self.ax = self.fig.add_subplot(121)
    self.ax_out = self.fig.add_subplot(122)
    if self.imshow == True:
      self.ax.imshow(self.data)
      self.ax_out.imshow(self.data)
    else:
      self.ax.plot(self.data)
      self.ax_out.plot(self.data)
    self.xs = []
    self.ys = []
    self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self)
    
    self.butw = 0.1
    self.buth = 0.05
    self.butx = 0.25
    self.buty = 0.85
    self.butsep = 0.15

    self.axb1 = plt.axes([self.butx, self.buty, self.butw, self.buth])
    self.but1 = Button(self.axb1,"Load Image")
    self.but1.on_clicked(self.loadimg)
    
    self.axb2 = plt.axes([self.butx+self.butsep, self.buty, self.butw, self.buth])
    self.but2 = Button(self.axb2,"Red Eye")
    self.but2.on_clicked(self.red_eye)
    
    self.axb3 = plt.axes([self.butx+2*self.butsep, self.buty, self.butw, self.buth])
    self.but3 = Button(self.axb3,"Motion Blur")
    self.but3.on_clicked(self.motion_blur)
    
    self.axb4 = plt.axes([self.butx+3*self.butsep, self.buty, self.butw, self.buth])
    self.but4 = Button(self.axb4,"Clear")
    self.but4.on_clicked(self.clear_img)
    

    plt.show()
    plt.close()

  def loadimg(self, event):
    root = Tkinter.Tk()
    root.withdraw()
    file_path = tkFileDialog.askopenfilename()
    root.destroy()
    tmp = cv2.imread(file_path)
    self.data = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    self.ax.clear()
    self.ax.imshow(self.data) 
    self.fig.canvas.draw()

  def red_eye(self, event):
    redimg = redI(self.data)
    self.ax_out.clear()
    self.ax_out.imshow(redimg) 
    self.fig.canvas.draw()

  def clear_img(self, event):
    self.ax_out.clear()

  def motion_blur(self, event):
    inpts = zip(self.ys,self.xs)
    bimg = mblur(inpts,self.data)
    self.xs = []
    self.ys = []
    self.ax_out.clear()
    self.ax_out.imshow(bimg) 
    self.fig.canvas.draw()
  
  def __call__(self, event):
    if event.inaxes!=self.ax.axes: return
    if event.button == 1:
      self.xs.append(int(np.around(event.xdata)))
      self.ys.append(int(np.around(event.ydata)))
    else:
      if not len(self.xs) == 0:
        self.xs.pop()
        self.ys.pop()
    xlim = self.ax.get_xlim()
    ylim = self.ax.get_ylim()
    self.ax.clear()
    if self.imshow == True:
      self.ax.imshow(self.data)
    else:
      self.ax.plot(self.data)
    for xpoint,ypoint in zip(self.xs,self.ys):
      self.ax.plot(
          [xpoint],
          [ypoint],
          color = 'red',
          marker='o',
          mfc='None')
    self.ax.set_xlim(xlim)
    self.ax.set_ylim(ylim)
    self.ax.figure.canvas.draw()

class RectangleBuilder:
  def __init__(self, data):
    self.data = data

    mainFig = plt.figure(figsize=(5,9))
    mainFig.canvas.set_window_title('Original image') 
    mainFig.suptitle("Use the mouse to drag a box around the area.\nThis area will be shown in the template figure.\nWhen you're happy with the subimage,\n close the two figures to continue the program.\nTo zoom, use the magnifying-glass button.\nTo go back to selection mode, press it again.")
    ax = mainFig.add_subplot(111)
    ax.imshow(self.data)
    self.rectSelc = RectangleSelector(ax, self.onselect, drawtype='box')

    self.x1 = None
    self.x2 = None
    self.y1 = None
    self.y2 = None
    self.subimage = None

    self.templateFig = plt.figure()
    self.templateFig.canvas.set_window_title('Subimage') 
    self.templatePlot = self.templateFig.add_subplot(111)
    plt.show()

    plt.close()

  def onselect(self, eclick, erelease):
    #Matplotlib and numpy defines x and y orthogonal
    #to eachother. Prompting this switch.
    self.y1, self.y2 = int(eclick.xdata), int(erelease.xdata)
    self.x1, self.x2 = int(eclick.ydata), int(erelease.ydata)
    self.subimage = self.data[
      self.x1:self.x2,
      self.y1:self.y2]
    self.templatePlot.imshow(self.subimage) 
    self.templateFig.canvas.draw()

def getSubImage(data):
  tempSubImage = RectangleBuilder(data)
  subImage = tempSubImage.subimage
  pos1 = tempSubImage.x1, tempSubImage.y1
  pos2 = tempSubImage.x2, tempSubImage.y2
  return(subImage, pos1, pos2)

def getPoints(data, imshow = True):
  tempPoints = PointsBuilder(data, imshow)
  #Matplotlib and numpy defines x and y orthogonal
  #to eachother. Prompting this switch.
  xpoints = tempPoints.ys
  ypoints = tempPoints.xs
  return(xpoints, ypoints)
