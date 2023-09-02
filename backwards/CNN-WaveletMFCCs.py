#training data prep

import numpy as np
import os
import glob
import pywt
import librosa
import subprocess
from scipy.fftpack import dct
import pickle

EXAMPLES_PER_BATCH = 32
WAVELET_NAME = "mexh"
WAVELET_SCALES = np.arange(0.5, 30, 0.5)  
SAMPLE_RATE = 24000
SAMPLES_PER_EXAMPLE = SAMPLE_RATE * 4
NUM_WAVES = 3
NUM_MELS = 64
NUM_OTHER_FEATURES = 11
NUM_DCT_COEFFICIENTS = 32
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

#compute a 'wavelet mel-filterbank'
def getFilterBank():
	freqs = [pywt.scale2frequency(WAVELET_NAME, scale) * SAMPLE_RATE for scale in WAVELET_SCALES] 

	def freq2Mel(freq):
		return 1125 * np.log(1 + freq / 700)

	lowestMel = freq2Mel(freqs[-1])
	highestMel = freq2Mel(freqs[0])
	mels = np.linspace(lowestMel, highestMel, NUM_MELS + 2)

	def mel2Freq(mel):
		return 700 * (np.exp(mel/1125) - 1)

	freqPoints = [mel2Freq(mel) for mel in mels]

	#get one row, lowest scale should be at the top of the CWT
	def melFilter(melNumber):
		row = np.zeros(len(WAVELET_SCALES))
		for i in range(len(WAVELET_SCALES)):
			currentScaleFreq = freqs[i]
			if currentScaleFreq >= freqPoints[melNumber]:
				if currentScaleFreq <= freqPoints[melNumber + 1]:
					row[i] = (currentScaleFreq - freqPoints[melNumber])/(freqPoints[melNumber + 1] - freqPoints[melNumber])
				elif currentScaleFreq <= freqPoints[melNumber + 2]:
					row[i] = (freqPoints[melNumber + 2] - currentScaleFreq )/(freqPoints[melNumber + 2] - freqPoints[melNumber + 1])
		return row

	return np.asarray([melFilter(i) for i in range(NUM_MELS)])

fbank = getFilterBank()
def processWavs(datapath):
	patchFiles = sorted(glob.glob(os.path.join(datapath, "*.txt")))
	waveFiles = sorted(glob.glob(os.path.join(datapath, "*.wav")))
	rawParams = [np.loadtxt(patchFile) for patchFile in patchFiles]
	waves = [librosa.load(waveFile, sr=SAMPLE_RATE, dtype=np.float32)[0] for waveFile in waveFiles]
	print(len(waves))
	print("calculating scaleograms")
	scaleograms = [pywt.cwt(wave, WAVELET_SCALES, WAVELET_NAME)[0] for wave in waves]
	print("applying mel filterbank")
	melScaleograms = [np.abs(np.matmul(fbank, scaleo))**2 for scaleo in scaleograms]
	print("converting to decibels")
	melLogScaleograms = [librosa.power_to_db(melScaleo) for melScaleo in melScaleograms]

	return (
		np.asarray([dct(mls, type=2, axis=1, norm="ortho")[:, :NUM_DCT_COEFFICIENTS] for mls in melLogScaleograms]),
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

	wavs = processWavs(datapath)

	file = open('./lastvalidation.pkl', 'wb')
	pickle.dump(wavs, file)
	file.close()
	#file = open('./lastvalidation.pkl', 'wb')
	#stuff = pickle.load(file)
	#file.close()
	return wavs #stuff

#defining the model
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, AveragePooling2D, Flatten, Softmax
from keras.losses import CategoricalCrossentropy, MeanSquaredError
def getModel():
	inputLayer = Input(shape=(NUM_MELS, NUM_DCT_COEFFICIENTS, 1))
	conv1 = Conv2D(64, (4,4), strides=(1,1), activation="sigmoid")(inputLayer)
	conv2 = Conv2D(128, (4,4), strides=(2,2), activation="relu")(conv1)
	conv3 = Conv2D(256, (4,4), strides=(2,2), activation="relu", padding="same")(conv2)
	#conv5 = Conv2D(256, (3,3), strides=(1,1), activation="relu", padding="same")(conv4)
	#flatten and then bifurcate the network into the classification and regression parts
	flat = Flatten()(conv3)
	print(flat.shape[1])
	classDense1 = Dense(min(flat.shape[1] // 2, NUM_WAVES * 4), activation="relu")(flat)
	classDense2 = Dense(min(flat.shape[1] // 4, NUM_WAVES * 2), activation="relu")(classDense1)
	classDense3 = Dense(NUM_WAVES, activation="relu")(classDense2)
	classOutput = Softmax(name="classout")(classDense3)
	regressionDense1 = Dense(min(flat.shape[1] // 4, NUM_OTHER_FEATURES * 4), activation="relu")(flat)
	regressionOutput = Dense(NUM_OTHER_FEATURES, name="regressionout", activation="sigmoid")(regressionDense1)

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