#training data prep

import numpy as np
import os
import glob
import pywt
import librosa
import subprocess
from scipy.fftpack import dct
import pickle
from random import sample

EXAMPLES_PER_BATCH = 32
WAVELET_NAME = "gaus4"
WAVELET_SCALES = np.arange(0.5, 32, 0.5)  
SAMPLE_RATE = 24000
SAMPLES_PER_EXAMPLE = SAMPLE_RATE * 4
NUM_WAVES = 3
NUM_MELS = 12
NUM_OTHER_FEATURES = 11
NUM_DCT_COEFFICIENTS = 6
NUM_FRAMES = 128
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
def getMelFilterBank():
	freqs = np.asarray(
		[pywt.scale2frequency(WAVELET_NAME, scale) * SAMPLE_RATE for scale in WAVELET_SCALES] 
	)

	def freq2Mel(freq):
		return 1125 * np.log(1 + freq / 700)

	def mel2Freq(mel):
		return 700 * (np.exp(mel/1125) - 1)

	lowestMel = freq2Mel(freqs[-1])
	highestMel = freq2Mel(freqs[0])
	mels = np.linspace(lowestMel, highestMel, NUM_MELS + 2)
	freqPoints = np.asarray([mel2Freq(mel) for mel in mels])

	differences = np.abs(freqPoints[:, np.newaxis] - freqs)
	minima = np.argmin(differences, axis=1)
	minima = np.flip(np.unique(minima))
	freqPoints = freqs[minima]

	def melFilterRow(melNumber):
		row = np.zeros(len(WAVELET_SCALES))
		for i in range(len(WAVELET_SCALES)):
			currentScaleFreq = freqs[i]
			if currentScaleFreq >= freqPoints[melNumber]:
				if currentScaleFreq <= freqPoints[melNumber + 1]:
					row[i] = (currentScaleFreq - freqPoints[melNumber])/(freqPoints[melNumber + 1] - freqPoints[melNumber])
				elif currentScaleFreq <= freqPoints[melNumber + 2]:
					row[i] = (freqPoints[melNumber + 2] - currentScaleFreq )/(freqPoints[melNumber + 2] - freqPoints[melNumber + 1])
			else:
				break
		return row

	fbank = np.asarray([melFilterRow(i) for i in range(len(freqPoints) - 2)])
	return fbank #/ np.sum(fbank, axis=1, keepdims=1)

def getDeltas(matrix):
	return np.c_[np.zeros(matrix.shape[0]), np.diff(matrix)]

def MFCCPlusDelta(wave):
	scaleo = np.abs(pywt.cwt(wave, WAVELET_SCALES, WAVELET_NAME)[0])
	melScaleo = np.matmul(fbank, scaleo)**2
	logMelScaleo = np.log(melScaleo + 1)
	cepstrum = dct(logMelScaleo, type=2, axis=1, norm="ortho")[:, :NUM_DCT_COEFFICIENTS]
	return np.hstack([cepstrum, getDeltas(cepstrum)])
	
frameMarkers = np.linspace(0, SAMPLES_PER_EXAMPLE, NUM_FRAMES, endpoint=False, dtype=int)[1:]
fbank = getMelFilterBank()
def processWavs(datapath):
	patchFiles = sorted(glob.glob(os.path.join(datapath, "*.txt")))
	waveFiles = sorted(glob.glob(os.path.join(datapath, "*.wav")))
	rawParams = [np.loadtxt(patchFile) for patchFile in patchFiles]
	waves = [librosa.load(waveFile, sr=SAMPLE_RATE, dtype=np.float32)[0] for waveFile in waveFiles]
	allFrames = [np.split(wave, frameMarkers) for wave in waves]
	#a list containing our samples each being a NUM_FRAMES x NUM_MELS X (NUM_DCT_COEFFICIENTS * 2)
	features = np.asarray([[MFCCPlusDelta(frame) for frame in framedSample] for framedSample in allFrames])
	return (
		features,
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

	#wavs = processWavs(datapath)

	#file = open('./lastvalidation.pkl', 'wb')
	#pickle.dump(wavs, file)
	#file.close()
	file = open('./lastvalidation.pkl', 'rb')
	wavs = pickle.load(file)
	file.close()
	return wavs

#defining the model
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, AveragePooling2D, Flatten, Softmax, BatchNormalization
from keras.losses import CategoricalCrossentropy, MeanSquaredError, MeanAbsolutePercentageError
from keras.backend import set_image_data_format, learning_phase

from keras import Model
from keras.losses import CategoricalCrossentropy, MeanSquaredError
class AdaptiveTraining(Model):
	def __init__(self, **kwargs):
		#self.catLoss = CategoricalCrossentropy(from_logits=False)
		#self.predLoss = MeanSquaredError()
		self.batchProbabilities = [
			np.full(NUM_WAVES, 1 / NUM_WAVES),
			np.full((NUM_OTHER_FEATURES, 10), 1 / (10 * NUM_OTHER_FEATURES))
		]

		set_image_data_format("channels_last")
		inputLayer = Input(shape=((NUM_FRAMES, fbank.shape[0], NUM_DCT_COEFFICIENTS * 2)))
		normalisation = BatchNormalization()(inputLayer)
		conv1 = Conv2D(48, (5,3), strides=(2,1), activation="relu", padding="valid")(normalisation)
		conv2 = Conv2D(64, (5,3), strides=(2,1), activation="relu", padding="valid")(conv1)
		conv3 = Conv2D(64, (5,3), strides=(2,1), activation="relu", padding="valid")(conv2)
		conv4 = Conv2D(96, (5,3), strides=(2,2), activation="relu", padding="valid")(conv3)
		#flatten and then bifurcate the network into the classification and regression parts
		flat = Flatten()(conv4)
		dense = Dense(flat.shape[1] * 4 // 5, activation="relu")(flat)
		classDense1 = Dense(max(flat.shape[1] * 2 // 3, NUM_WAVES * 4), activation="relu")(dense)
		classDense2 = Dense(NUM_WAVES, activation="relu")(classDense1)
		classOutput = Softmax(name="classout")(classDense2)
		regressionDense1 = Dense(max(flat.shape[1] * 2 // 3, NUM_OTHER_FEATURES * 4), activation="relu")(dense)
		regressionOutput = Dense(NUM_OTHER_FEATURES, name="regressionout", activation="relu")(regressionDense1)
		super(AdaptiveTraining, self).__init__(
			**kwargs,
			inputs = [inputLayer],
			outputs = [classOutput, regressionOutput]
		)

	def validation_step(self, data):
		x, y = data.numpy()
		yPred = self(x, training=False)
		print(yPred)

		categoricalLosses = np.sum(- y[0] * np.log(yPred[0]), axis=0)
		categoricalLoss = np.sum(categoricalLosses)
		self.batchProbabilities[0] = categoricalLosses / categoricalLoss
		print(f"categorical loss: {self.batchProbabilities[0]}")

		squaredErrors = (y[1] - yPred[1]) ** 2
		binFeatureErrors = np.apply_along_axis(
			lambda x: np.bincount(np.floor(10 * x).astype(np.int64), minlength=10),
			1,
			squaredErrors.T
		)
		self.batchProbabilities[1] = binFeatureErrors / np.sum(binFeatureErrors, axis=1)[:, np.newaxis]
		print(f"continuous loss: {self.batchProbabilities[1]}")

		# Update the metrics.
		for metric in self.metrics:
			if metric.name != "loss":
				metric.update_state(y, yPred)
				
		return super().test_step(self, data) #{m.name: m.result() for m in self.metrics}

	def test_step(self, data):
		if learning_phase() == 1:
			print("learning")
			return super().test_step(self, data)
		else:
			print("validating")
			return self.validation_step(data)

def getModel():
	model = AdaptiveTraining()
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
	#model.load_weights("C:\\Users\\abdulg\\Desktop\\waves\\checkpoint - Copy.keras")
	model.summary()
	validationData = generateValidationSet()
	
	history = {}
	while True:
		history = model.fit(
			x=generateBatch(), #yields a generator
			validation_data=validationData,
			callbacks=[stoppingCondition, checkpoint],
			epochs=3,
			#steps_per_epoch = 32,
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
	run()