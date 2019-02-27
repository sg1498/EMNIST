from tkinter import *
import matplotlib.pyplot as plt
from PIL import ImageFilter,Image
import numpy as np
#import cv2
from IPython import get_ipython
import pyperclip

def input_emnist(st):
	#opening the input image to be predicted
    im_open = Image.open(st)
    im = Image.open(st).convert('LA') #conversion to gray-scale image
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L',(28,28),(255))


    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((28.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((28,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (0,wtop)) #paste resized image on white canvas
    else:
    #Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((28.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
     # resize and sharpen
        img = im.resize((nwidth,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #calculate vertical pozition
        newImage.paste(img, (wleft,0)) #paste resize


# # Normalizing image into pixel values



    tv = list(newImage.getdata())
    tva = [ (255-x)*1.0/255.0 for x in tv]



    for i in range(len(tva)):
        if tva[i]<=0.45:
            tva[i]=0.0
    n_image = np.array(tva)
    rn_image = n_image.reshape(28,28)
    #displaying input image
    plt.imshow(im_open)
    plt.title("Input Image")
    plt.show()
    #displaying gray-scale image
    plt.imshow(newImage.convert('LA'))
    plt.title("Rescaled Image")
    plt.show()
    #displaying normalized image
    plt.imshow(n_image.reshape(28,28))
    plt.title("Normalized Image")
    plt.show()
    # return all the images
    return n_image,im_open,newImage

'''

from keras.models import Sequential
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dense
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        #model=Sequential()
        model.add(Conv2D(16, kernel_size=4, input_shape=inputShape, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=4, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2))
        model.add(Conv2D(64, kernel_size=4, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=4, activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(3200, activation='tanh'))
        model.add(BatchNormalization())
        model.add(Dense(47, activation='softmax'))
        #model.summary()
        
        return model
 '''
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model=Sequential()
        model.add(Conv2D(128, kernel_size=3, input_shape=inputShape, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2))
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        
        return model   

import tensorflow as tf 
def model_predict(n_image):
    
    model = LeNet.build(width=28, height=28, depth=1, classes=47)
#model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    from skimage import transform,io
#gray = io.imread('file name with path', as_gray = True)
    model.load_weights('emnist.pt')
    arr=n_image.reshape(1,28,28,1)
    
    prediction = model.predict(arr)[0]
    #print(prediction)
    pred = np.argmax(prediction)
    
    labels_dict ={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'l',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'u',31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'d',39:'e',40:'f',41:'g',42:'h',43:'n',44:'q',45:'r',46:'t',47:'அ',48:'ஆ',49:'இ',50:'ஈ',51:'உ',52:'ஊ',53:'எ',54:'ஏ',55:'ஐ',56:'ஒ',57:'ஓ',58:'ஔ'}
    
    s = "The predicted character is {}".format(labels_dict[pred])
    print(s)
    print('pred :', pred)
    return s,labels_dict[pred]












st = ""
root = Tk() #tkinter GUI
root.geometry("500x500")
root.winfo_toplevel().title("Handwritten Character Recognition")
label1 = Label( root, text="Enter the name of the file: ")
E1 = Entry(root, bd =5)
def getVal():
    global st
    st= E1.get()
    root.destroy()
submit = Button(root, text ="Submit", command = getVal)
label1.pack()
E1.pack()
submit.pack(side =BOTTOM)
mainloop()
n_image,image,convo_image = input_emnist(st) #call to Function with name of the file as parameter
res,cpy = model_predict(n_image)
pyperclip.copy(str(cpy)) #copy the predicted character to clipboard
root2 = Tk()
root2.winfo_toplevel().title("Handwritten Character Recognition")
root2.geometry("500x500")
label2 = Label(root2,text = res)
label2.config(font=("Courier", 20))
label2.pack()
mainloop()
