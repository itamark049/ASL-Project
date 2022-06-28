from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os


class Use_Model:
    def __init__(self, path):
        self.capitals = ['B','F','H','I']
        self.match_labels = {'B':0,'F':1,'H':2,'I':3}
        self.model = Sequential()
        self.x_train = 0 
        self.y_train = 0 
        self.x_test = 0 
        self.y_test = 0 
        data, labels = self.load_dataset(path)
        self.build_model()
        self.split(data, labels)
        
    def convert_into_category(self, character):
        """
        
        converts letter the image is from to the correct label in category
        Parameters
        ----------
        character : sttring
            Letter to covert.

        Returns
        -------
        ls : list
            list with all 0 and 1 in the position of the label.

        """
        ls=[0 for i in range(4)]
        ls[self.match_labels[character]]=1
        return ls
    def load_dataset(self,path):
        """
        loads images from path to data and labels 

        Parameters
        ----------
        path : string
            path to dataset.

        Returns
        -------
        data : list
            list of all images loaded with Keras.
        labels : list
            labels of the images.

        """
        data=[]
        labels=[]
        for category in self.capitals:
            directory=os.path.join(path,category)
            print(directory)
            for img in os.listdir(directory):
                img_path=os.path.join(directory,img)
                image=load_img(img_path,target_size=(128,128),color_mode="grayscale") #Height and width of images
                image=img_to_array(image)

                data.append(image)
                labels.append(self.convert_into_category(character = category))

        data=np.array(data,dtype="float32")
        labels=np.array(labels)
        return data, labels

    def build_model(self):
        """
        

        Builds model with 12 hidden layers, categorical crossentropy loss function and Adam optimizer

        """

        self.model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (128,128,1)))
        self.model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
        self.model.add(Dropout(0.3))
        
        self.model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu' ))
        self.model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
        self.model.add(Dropout(0.3))
        
        self.model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu' ))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same',activation ='relu'))
        
        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))

        self.model.add(Activation('relu'))
        self.model.add(Dense(64))
        self.model.add(Dropout(0.5))

        self.model.add(Activation('relu'))
        self.model.add(Dense(32))
        self.model.add(Dropout(0.5))

        self.model.add(Activation('relu'))
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dense(4, activation = "softmax"))

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    def split(self, data, labels):
        """
        

        Parameters
        ----------
        data : list of images
            all images loaded from dataset
        labels : list of labels
            list of all the labels with index corresponding to index in data list.

        

        """
        (self.x_train,self.x_test,self.y_train,self.y_test)=train_test_split(data,labels,test_size=0.1)
            
        self.x_train = np.array(self.x_train) / 255
        self.x_test = np.array(self.x_test) / 255
        
        self.x_train.reshape(-1, 128,128, 1)
        self.y_train = np.array(self.y_train)
        
        self.x_test.reshape(-1, 128,128, 1)
        self.y_test = np.array(self.y_test)
        
        
    def train_model(self):
         """
        Train model on x_train and y_Train

        Returns
        -------
        history : keras model history
            Train and Loss records for each epoch.

        """
         history=self.model.fit(self.x_train,self.y_train, epochs=15 , batch_size=32, validation_split=0.10,)
         self.model.save("my model4")
         return history
    
    
    def test_images(self):
        """
        Tests model on x_test and y_test, returns string which shows results. 
        """
        results = self.model.evaluate(self.x_test, self.y_test, batch_size=32)
        return "test loss: {}, test acc: {}".format(results[0], results[1])

def main():
    x = Use_Model(r'C:\Users\itama\Documents\dataset')
    history = x.train_model()
    x.test_images()
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
if __name__=="__main__":
    main()