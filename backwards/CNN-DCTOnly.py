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
WAVELET_SCALES = np.arange(0.5, 32, 2)  
SAMPLE_RATE = 24000
SAMPLES_PER_EXAMPLE = SAMPLE_RATE * 4
NUM_WAVES = 3
NUM_OTHER_FEATURES = 11
NUM_DCT_COEFFICIENTS = 1200
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

def unnormaliseParams(params):
	return params * PARAMETER_RANGE + PARAMETER_LBS

def processWavs(datapath):
	patchFiles = sorted(glob.glob(os.path.join(datapath, "*.txt")))
	waveFiles = sorted(glob.glob(os.path.join(datapath, "*.wav")))
	waves = [librosa.load(waveFile, sr=SAMPLE_RATE, dtype=np.float32)[0] for waveFile in waveFiles]
	scaleograms = [pywt.cwt(wave, WAVELET_SCALES, WAVELET_NAME)[0] for wave in waves]
	rawParams = [np.loadtxt(patchFile) for patchFile in patchFiles]

	return (
		np.asarray([dct(scaleo, type=2, axis=1)[:, :NUM_DCT_COEFFICIENTS] for scaleo in scaleograms]),
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
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Softmax
from keras.losses import CategoricalCrossentropy, MeanSquaredError
def getModel():
	inputLayer = Input(shape=(len(WAVELET_SCALES), NUM_DCT_COEFFICIENTS, 1))
	conv1 = Conv2D(16, (3,4), strides=(2,3), activation="relu")(inputLayer)
	pool1 = MaxPooling2D(pool_size=(1,3), strides=(1,2), padding="same")(conv1)
	conv2 = Conv2D(32, (3,4), strides=(2,3), activation="relu")(pool1)
	pool2 = MaxPooling2D(pool_size=(1,3), strides=(1,2), padding="same")(conv2)
	conv3 = Conv2D(64, (3,3), strides=(2,2), activation="relu", padding="same")(pool2)
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
def train():
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

def evaluate(wav):
	wave = librosa.load(wav, sr=SAMPLE_RATE, dtype=np.float32)[0]
	cwt = pywt.cwt(wave, WAVELET_SCALES, WAVELET_NAME)[0]
	dctCoeffs = dct(cwt, type=2, axis=1)[:, :NUM_DCT_COEFFICIENTS]

	model = getModel()
	print(model.summary())
	model.load_weights("C:\\Users\\abdulg\\Desktop\\waves\\model.keras")
	[categoryProbs, estimatedParams] = model.predict(np.asarray([dctCoeffs]))
	predictedCategory = np.argmax(categoryProbs[0])

	category = np.zeros(NUM_WAVES)
	category[predictedCategory] = 1

	return np.concatenate([category, estimatedParams[0]])


def formatPrediction(prediction):
	return ("{:.2f} " * len(prediction)).format(*unnormaliseParams(prediction))

if __name__ == "__main__":
	#print("running")
	#model = train()
	#model.save("C:\\Users\\abdulg\\Desktop\\waves\\attempt1.keras")

	print(formatPrediction(evaluate("C:\\Users\\abdulg\\source\\repos\\Synth\\backwards\\validationdata\\78.wav")))