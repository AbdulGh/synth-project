#training data prep

import numpy as np
import os
import glob
import pywt
import librosa
import subprocess
from scipy.fftpack import dct

EXAMPLES_PER_BATCH = 32
WAVELET_NAME = "mexh"
WAVELET_SCALES = np.arange(1, 30)  
SAMPLE_RATE = 16000
SAMPLES_PER_EXAMPLE = 64000
NUM_WAVES = 3
NUM_OTHER_FEATURES = 9
NUM_DCT_COEFFICIENTS = 1024
SYNTHESIZER_PATH = "C:\\Users\\abdulg\\source\\repos\\Synth\\out\\build\\x64-debug\\synth.exe"

def processWavs(datapath):
	patchFiles = sorted(glob.glob(os.path.join(datapath, "*.txt")))
	waveFiles = sorted(glob.glob(os.path.join(datapath, "*.wav")))
	waves = [librosa.load(waveFile, sr=SAMPLE_RATE, dtype=np.float32)[0] for waveFile in waveFiles]
	scaleograms = [pywt.cwt(wave, WAVELET_SCALES, WAVELET_NAME)[0] for wave in waves]
	return (
		np.asarray([dct(scaleo, type=2, axis=1)[:, :NUM_DCT_COEFFICIENTS] for scaleo in scaleograms]),
		np.split( #split to separate the categorisation from the rest
			np.asarray([np.loadtxt(patchFile) for patchFile in patchFiles]),
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
		subprocess.run(f"{SYNTHESIZER_PATH} {EXAMPLES_PER_BATCH} {datapath}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		yield processWavs(datapath)

def generateValidationSet():
	print("generating validation data")
	datapath = os.path.abspath("./validationdata")
	#subprocess.run(f"{SYNTHESIZER_PATH} 200 {datapath}", stdout=subprocess.DEVNULL)
	return processWavs(datapath)

#defining the model
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Softmax
from keras.losses import CategoricalCrossentropy, MeanSquaredError
def getModel():
	inputLayer = Input(shape=(len(WAVELET_SCALES), NUM_DCT_COEFFICIENTS, 1))
	conv1 = Conv2D(16, (2,3), activation="relu")(inputLayer)
	pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding="valid")(conv1)
	conv2 = Conv2D(32, (2,2), activation="relu")(pool1)
	pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding="valid")(conv2)
	conv3 = Conv2D(64, (2,2), activation="relu")(pool2)
	#flatten and then bifurcate the network into the classification and regression parts
	flat = Flatten()(conv3)
	classDense1 = Dense(min(flat.shape[1] // 4, NUM_WAVES * 4), activation="relu")(flat)
	classDense2 = Dense(NUM_WAVES, activation="relu")(classDense1)
	classOutput = Softmax(name="classout")(classDense2)
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

from keras.callbacks import EarlyStopping
import pickle
def run():
	#the goal is even to 'overfit' the generator, but we still could do with a stopping condition
	stoppingCondition = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
	model = getModel()
	
	history = {}
	while True:
		history = model.fit(
			x=generateBatch(), #yields a generator
			validation_data=generateValidationSet(),
			callbacks=[stoppingCondition],
			epochs=1,
			steps_per_epoch = 32,
			verbose=1
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