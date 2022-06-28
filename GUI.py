
from tkinter import filedialog as fd
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from zipfile import ZipFile
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
from MODEL import Use_Model
def find(name):
    """
    finds file in current working directory

    Parameters
    ----------
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    string
        path to file if it was found.

    """
    for root, dirs, files in os.walk(os.getcwd()):
        if name in files:
            return os.path.join(root, name)

class create_extracting_GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root['bg'] = "#080064"
    
   
    

    def input_from_window(self):
        """
        makes user choose zip file, and extracts all files from it unless it sees files have already been extracted

        Returns
        -------
        None.

        """
        self.path =  fd.askopenfilename(title='Open a file',initialdir='/') #contains path to the file the user picks
        while (".zip" not in self.path):
            mb.showwarning("wrong", 'choose zip file')
            self.path =  fd.askopenfilename(title='Open a file',initialdir='/')
        
        with ZipFile(self.path,'r') as zp:
                if(os.path.isdir('Itamar Project')):
                    mb.showwarning("Error", "Already extracted")
                    self.root.destroy()
                else:
                    zp.extractall()
                    mb.showinfo("extract", 'extracted successfully')
        self.root.destroy()
    
    def create(self):
        """
        creates window with size 300x300 with desired rows and columns

        
        """
        self.root.resizable(True, True)
        self.root.title("Extract files")
        self.root.geometry('300x300+50+50')
        self.root.columnconfigure(0, weight = 1)
        
        self.root.rowconfigure(0,weight = 1)
        self.root.rowconfigure(1, weight = 1)

        
        
        already_extracted = tk.Button(self.root, text = "Already Extracted",  bg= "red" , command=lambda: self.root.destroy())
        already_extracted.grid(column=0, row=0)
        
        extract_button= tk.Button(self.root, text = "Extract Files",  bg= "green" , command = lambda: self.input_from_window())
        extract_button.grid(column=0, row=1)
        self.root.mainloop()

class create_Test_GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root['bg'] = '#080064'
    def create_exit_button(self):
        """
        creates window's exit button

      
        """
        exit_button = tk.Button(self.root, text = "Exit",  bg= "red" , command=lambda: self.root.destroy())
        exit_button.grid(column=0, row=1)
    
    
    def test_on_image(self):
        """
        Makes user pick an image and runs model on it, then shows message box with prediction and condifence. 

        

        """
        model = keras.models.load_model(r"Itamar Project\model") # trained model extracted from zip file
        filename =  fd.askopenfilename(title='Open a file',initialdir='/',filetypes=(('jpeg files', '*.jpg'),('jpeg files', '*.jpeg')))
        #contains path to image the user picked. 
        if("jpg" not in filename and "jpeg" not in filename):
            tk.messagebox.showwarning("wrong", "didn't choose correct file type")
        else:
            capitals = ['B','F','H','I'] # list of letters model is trained on, used to show on screen the prediction the model made on image
            img = load_img(filename,target_size=(128,128),color_mode="grayscale")#mask image
            img_array = img_to_array(img)
            img = img_array / 255.0 #normalize
            img = np.expand_dims(img, 0) 
            score = model.predict(img)
            predicted, confidence = capitals[np.argmax(score[0])], 100 * np.max(score[0])
            tk.messagebox.showinfo("prediction", "This image most likely belongs to {} with a {:.2f} percent confidence.".format(predicted,confidence))
    
            
    def create(self):
        """
        creates window with size 200x200 with desired rows and columns

        
        """
        self.root.resizable(True, True)
        self.root.title("Test images")
        self.root.geometry('200x200+50+50')
        self.root.columnconfigure(0, weight = 1)
        
        self.root.rowconfigure(0,weight = 1)
        self.root.rowconfigure(1, weight = 1)

        
        self.create_exit_button()
        testing = tk.Button(self.root, text = "test",  bg= "#01C5FF" , command= self.test_on_image)
        testing.grid(column=0, row=0)
        self.root.mainloop()

class create_main_GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root['bg'] = '#080064'
        self.train_button = tk.Button(self.root, text = "Train",bg = "#01C5FF",  command = self.train_pressed, state= DISABLED)
        self.test_button = tk.Button(self.root, text = "Test",  bg= "#01C5FF" , command = self.test_pressed, state = DISABLED)

    def add_image(self, name, size):
        """
        puts image on screen

        Parameters
        ----------
        path : string
            path to image we put on screen.
        size : list of int
            size of image we put on screen.



        """
        path = find(name)
        if(os.path.exists(path)):
            im = Image.open(path)
            im = im.resize(size)
            ph = ImageTk.PhotoImage(im, master = self.root) # image saved in way fit to show on tkinter window
            image_label = ttk.Label(self.root, image=ph)
            image_label.image=ph
            image_label.grid(column=2, row=0)
        
    def create_exit_button(self):
        """
        

        creates button that when pressed closes window
        """
        exit_button = tk.Button(self.root, text = "Exit",  bg= "red", command=lambda: self.root.destroy())
        exit_button.grid(column=2, row=4)
    
    def train_pressed(self):
        """
        

        Hides window. Then, loads data and train model while showing messages of current state. In the end it unhides the 
        window then shows accuracy and loss graph

        """
        self.root.withdraw()
        mb.showinfo("showinfo", "loading images!")
        trainer = Use_Model(path = r'Itamar Project\dataset') 
        mb.showinfo("showinfo", "starting to train images")
        history = trainer.train_model()
        self.root.deiconify()
        self.graph(history)
    
    def graph(self, history):
        """
        Plots accuracy and loss graphs

        Parameters
        ----------
        history : keras model history
            

        
        """
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
    def test_pressed(self):
        """
        Called when test button is pressed, 

        Creates test window.
        """
        window_test = create_Test_GUI()
        window_test.create()
    
    def extract_pressed(self):
        """
        Called when extract button is pressed, 

        Changes train & test button states to normal then creates extract window.
        """
        window_extract = create_extracting_GUI()
        self.train_button['state'] = NORMAL
        self.test_button['state'] = NORMAL
        window_extract.create()
    def buttons(self):
        """
        Creates buttons
        """
        #create train button
        self.train_button.grid(column=2, row=3)
        
        #create extract button
        extract = tk.Button(self.root, text = "Extract",  bg= "#01C5FF",  command = self.extract_pressed)
        extract.grid(column=2, row=1)
        
        #create test button
        self.test_button.grid(column=2, row=2)
   
    def create(self):
        """
        

        creates window with size 800x650 with desired rows and columns

        """
        self.root.resizable(True, True)
        self.root.title("hand gesture letters")
        self.root.geometry('800x650+50+50')
        self.root.columnconfigure(0, weight = 1)
        self.root.columnconfigure(1, weight = 1)
        self.root.columnconfigure(2, weight = 1)
        self.root.columnconfigure(3, weight = 1)
        self.root.columnconfigure(4, weight = 1)
        
        self.root.rowconfigure(0,weight = 1)
        self.root.rowconfigure(1, weight = 1)
        self.root.rowconfigure(2, weight = 1)
        self.root.rowconfigure(3, weight = 1)
        self.root.rowconfigure(4, weight = 1)
        
    def open(self):
        self.root.mainloop()
        

#shape window
def main():
    
    window = create_main_GUI()
    window.create()
    window.add_image(name = 'asl.png', size = (400,400))
    window.create_exit_button()
    window.buttons()
    window.open()
if (__name__=="__main__"):
    main()
