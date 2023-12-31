#training data prep

import numpy as np
import os
import glob
import pywt
import librosa
import subprocess

EXAMPLES_PER_BATCH = 32
WAVELET_NAME = "mexh"
WAVELET_SCALES = np.arange(0.5, 32, 0.5)  
SAMPLE_RATE = 24000
SAMPLES_PER_EXAMPLE = SAMPLE_RATE * 4
NUM_WAVES = 3
NUM_OTHER_FEATURES = 11
SYNTHESIZER_PATH = "C:\\Users\\abdulg\\source\\repos\\Synth\\out\\build\\x64-debug\\synth.exe"

"""
continuous parameters currently look like
pitch
filterCutoff
filterResonance
fAttackTime
fDecayTime
fSustainLevel
fReleaseTime
fModFreq
fModInt
pModInt (vibrato)
pModFreq
"""

#for normalization
PARAMETER_LBS = np.asarray([0, 0, 0, 440, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0])
PARAMETER_RANGE = np.asarray([1, 1, 1, 1661.22, 3322, 1, 1, 1, 1, 1, 8, 0.5, 8, 0.02]) - PARAMETER_LBS

def normaliseParams(params):
	return (params - PARAMETER_LBS)/PARAMETER_RANGE

def processWavs(datapath):
	patchFiles = sorted(glob.glob(os.path.join(datapath, "*.txt")))
	waveFiles = sorted(glob.glob(os.path.join(datapath, "*.wav")))
	waves = [librosa.load(waveFile, sr=SAMPLE_RATE, dtype=np.float32)[0] for waveFile in waveFiles]
	scaleograms = [pywt.cwt(wave, WAVELET_SCALES, WAVELET_NAME)[0] for wave in waves]
	rawParams = [np.loadtxt(patchFile) for patchFile in patchFiles]

	return (
		np.asarray(scaleograms),
		np.split( #split to separate the categorisation from the rest
			[normaliseParams(params) for params in rawParams],
			[NUM_WAVES],
			axis=1
		)
	)

#generate a batch of waves, return their (compressed) scaleograms and features
def generateBatch():
	print("generating new batch")
	datapath = os.path.abspath("./trainingdata")
	while True:
		#will overwrite all of the previous batch as examplesPerBatch stays constant
		#todo - figure out why STK writes a newline to stderr per written file
		subprocess.run(f"{SYNTHESIZER_PATH} {EXAMPLES_PER_BATCH} {datapath}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		yield processWavs(datapath)

def generateValidationSet():
	print("generating validation data")
	datapath = os.path.abspath("./validationdata")
	#subprocess.run(f"{SYNTHESIZER_PATH} 600 {datapath}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	return processWavs(datapath)

#defining the model
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, AveragePooling2D, Flatten, Softmax
from keras.losses import CategoricalCrossentropy, MeanSquaredError
def getModel():
	inputLayer = Input(shape=(len(WAVELET_SCALES), SAMPLES_PER_EXAMPLE, 1))
	pool0 = AveragePooling2D(pool_size=(1,20), strides=(1,20), padding="same")(inputLayer)
	conv1 = Conv2D(16, (3,8), strides=(2,6), activation="relu")(pool0)
	pool1 = AveragePooling2D(pool_size=(1,8), strides=(1,7), padding="same")(conv1)
	conv2 = Conv2D(32, (3,7), strides=(1,6), activation="relu")(pool1)
	pool2 = AveragePooling2D(pool_size=(1,5), strides=(1,4), padding="same")(conv2)
	conv3 = Conv2D(64, (2,3), strides=(2,2), activation="relu", padding="same")(pool2)
	conv4 = Conv2D(128, (2,2), strides=(1,1), activation="relu", padding="same")(conv3)
	#flatten and then bifurcate the network into the classification and regression parts
	flat = Flatten()(conv4)
	print(flat.shape[1])
	classDense1 = Dense(min(flat.shape[1] // 2, NUM_WAVES * 4), activation="relu")(flat)
	classDense2 = Dense(min(flat.shape[1] // 4, NUM_WAVES * 2), activation="relu")(classDense1)
	classDense3 = Dense(NUM_WAVES, activation="relu")(classDense2)
	classOutput = Softmax(name="classout")(classDense3)
	regressionDense1 = Dense(min(flat.shape[1] // 4, NUM_OTHER_FEATURES * 4), activation="relu")(flat)
	regressionOutput = Dense(NUM_OTHER_FEATURES, name="regressionout", activation="relu")(regressionDense1)

	model = Model(
		inputs = [inputLayer],
		outputs = [classOutput, regressionOutput],
	)

	model.compile(
		optimizer="adam",
		loss={
			"classout": CategoricalCrossentropy(from_logits=False),
			"regressionout": MeanSquaredError(),
		},
		metrics=["accuracy"]
	)

	return model

from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
def run():
	checkpoint = ModelCheckpoint(
		"C:\\Users\\abdulg\\Desktop\\waves\\checkpoint.keras",
		save_weights_only=True,
		save_best_only=True,
		monitor="val_loss",
		mode="min",
		save_freq="epoch"
	)
	#the goal is even to 'overfit' the generator, but we still could do with a stopping condition
	stoppingCondition = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
	model = getModel()
	validationData = generateValidationSet()
	
	history = {}
	while True:
		history = model.fit(
			x=generateBatch(), #yields a generator
			validation_data=validationData,
			callbacks=[stoppingCondition, checkpoint],
			epochs=1,
			steps_per_epoch = 32,
			batch_size=32,
			verbose=2
		)

		if stoppingCondition.stopped_epoch > 0:
			break

	with open("./lastHistory", "wb") as histFile:
		pickle.dump(history.history, histFile)

	return model

if __name__ == "__main__":
	print("running")
	model = run()
	model.save("C:\\Users\\abdulg\\Desktop\\waves\\attempt1.keras")