# -*- coding: utf-8 -*-



# Import libraries
# .................

import os
import librosa   #for audio processing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from scipy.io import wavfile #for audio processing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

warnings.filterwarnings("ignore")


# ...........................................................................

                                        # basic setup

audioPth        = 'speechsub'
resmpRate       = 8000
inputLength     = 8000
modelname       = 'speechRV1'

                            # Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'




# ............................................................................


                                        # Plot a 'go' soundwave

goSample    = os.path.join(audioPth,
                           'go',
                           '00f0204f_nohash_1.wav')
(smp,smpR)  = librosa.load(goSample,sr=16000)


plt.figure(figsize=(8,4))
plt.plot(np.linspace(0,                 # Start
                     len(smp)/smpR,     # Stop (convert sample number into second)
                     len(smp)),         # Number of points
         smp)
plt.title('A sound of "go" ...')
plt.xlabel('time in seconds')
plt.ylabel('amplitude')



# ............................................................................

print("Inspecting data  ... ")
                                            # Inspect the data set

labels      = ["go",
               "stop",
               "yes", 
               "no", 
               "up", 
               "down", 
               "left", 
               "right", 
               "on", 
               "off"]

                                            # Check the number of records available in
                                            # each category
numOfRecords= []

for lbl in labels:
    pth     = os.path.join(audioPth,lbl)    # Get the path (the directory) for each label
    records = [f for f in os.listdir(pth) if f.endswith('.wav')]
                                            # Get all the wave files in each directory
    
    numOfRecords.append(len(records))       # Get the number of wave files in each directory
    
    
plt.figure(figsize=(8,8))
plt.barh(np.arange(len(labels)), 
         numOfRecords,
         color="C2")
plt.xlabel('No. of recordings')
plt.ylabel('Commands')
plt.yticks(np.arange(len(labels)), labels)
plt.title('The dataset')
plt.show()



                                            # Check distribution of durations in 
                                            # the dataset
durations   = []

for lbl in labels:
    pth     = os.path.join(audioPth,lbl)    # Get the path (the directory) for each label
    records = [f for f in os.listdir(pth) if f.endswith('.wav')]
                                            # Get all the wave files in each directory
    
    for rcd in records:
        (smpR,smp)  = wavfile.read(os.path.join(pth,rcd))
        
        durations.append(float(len(smp)/smpR))      
                                            # Get the duration for each wave file

plt.figure(figsize=(8,8))
plt.title('The distribution of duration')
plt.ylabel('The number of recordings')
plt.xlabel('Time in seconds')
durationHist    = plt.hist(durations)





# ............................................................................


                                            # Prepare dataset for training 
                                            # (Get only sample with a duration 
                                            # of 1 second)
print("Preparing dataset ... ")
allRecords  = []
allLabels   = []


for lbl in labels:
    pth     = os.path.join(audioPth,lbl)    # Get the path (the directory) for each label
    records = [f for f in os.listdir(pth) if f.endswith('.wav')]

    for rcd in records:
        (smp,smpR)  = librosa.load(os.path.join(pth,rcd),sr=16000)
        smp         = librosa.resample(smp,
                                       smpR, 
                                       resmpRate)
        if (len(smp)==inputLength): 
            allRecords.append(smp)
            allLabels.append(lbl)
            
            
allRecords  = np.array(allRecords).reshape(-1,inputLength,1)           




# ............................................................................


                                            # Prepare the label for training
                                            
le      = LabelEncoder()
lbls    = le.fit_transform(allLabels)
classes = list(le.classes_)                 # the output is a funny numpy str_ object
classes = [str(c) for c in classes]         # convert each output in the list to string
lbls    = to_categorical(lbls,num_classes=len(classes)) 




# ............................................................................

                                            # Split the data into training and 
                                            # validation set

(trDat,
 vlDat,
 trLbl, 
 vlLbl) = train_test_split(allRecords,
                           lbls,
                           stratify=lbls,
                           test_size=0.2,
                           random_state=229,
                           shuffle=True)



# ...........................................................................


                                            # Create the deep learning model
def createModel(inputSize):
    ipt = Input(shape=(inputSize,1))

    x   = Conv1D(8, 11, padding='valid', activation='relu')(ipt)
    x   = MaxPooling1D(4)(x)
    x   = Dropout(0.25)(x)
    
    x   = Conv1D(16, 11, padding='valid', activation='relu')(x)
    x   = MaxPooling1D(4)(x)
    x   = Dropout(0.25)(x)
    
    x   = Conv1D(32, 11, padding='valid', activation='relu')(x)
    x   = MaxPooling1D(4)(x)
    x   = Dropout(0.25)(x)
    
    x   = Conv1D(64, 11, padding='valid', activation='relu')(x)
    x   = MaxPooling1D(4)(x)
    x   = Dropout(0.25)(x)
    
    x   = Flatten()(x)
    x   = Dense(256, activation='relu')(x)
    x   = Dropout(0.5)(x)
    
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.5)(x)
    
    x   = Dense(len(classes), activation='softmax')(x)
    
    model = Model(ipt, x)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model



                                        # Setup the models
model       = createModel(inputLength) # This is meant for training
modelGo     = createModel(inputLength) # This is used for final testing

model.summary()

plot_model(model, 
           to_file=modelname+'_plot.pdf', 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')



# .............................................................................


                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_loss', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='min')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger]



# .............................................................................


                            # Fit the model
                            # This is where the training starts
model.fit(trDat, 
          trLbl, 
          validation_data=(vlDat, vlLbl), 
          epochs=100, 
          batch_size=32,
          shuffle=True,
          callbacks=callbacks_list)



# ......................................................................


                            # Now the training is complete, we get
                            # another object to load the weights
                            # compile it, so that we can do 
                            # final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])

 




# .......................................................................


                            # Make classification on the test dataset
predicts    = modelGo.predict(vlDat)


                            # Prepare the classification output
                            # for the classification report
predout     = np.argmax(predicts,axis=1)
testout     = np.argmax(vlLbl,axis=1)



testScores  = metrics.accuracy_score(testout,predout)
confusion   = metrics.confusion_matrix(testout,predout)


print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=classes,digits=4))
print(confusion)


    
    
    
# ..................................................................
    


records     = pd.read_csv(modelname +'.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0.00,0.40,0.60,0.80])
plt.title('Loss value',fontsize=12)

ax          = plt.gca()
ax.set_xticklabels([])



plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6,0.7,0.8,0.9])
plt.title('Accuracy',fontsize=12)
plt.show()



# ............................................................................

                                            # Create a function to make prediction 
                                            # straightly from a wave file
                                            
def commandPred(file):
    (smp,smpR)  = librosa.load(file,sr=16000)
    smp         = librosa.resample(smp,
                                   smpR, 
                                   resmpRate)
    smp         = smp.reshape(-1,inputLength,1)
    pred        = modelGo.predict(smp)
    pred        = np.argmax(pred,axis=1)
    
    return classes[pred[0]]


# ............................................................................
    
wfile       = 'voice01.wav'
pred        = commandPred(wfile)
print("")
print("The command predicted from '%s' is '%s'." % (wfile,pred) )