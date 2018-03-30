import numpy as np
import csv
import time
from scipy import stats
from scipy.signal import argrelmax

# from sklearn import svm
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

##########################################################################################

feature_dict = {}
dataset = []
GSRSlopes = []
meanGSR = [] 
maxGSR = []
minGSR = []
numGSRPeaks = []
meanGSRPeaks = []
q25GSRPeaks = []
q50GSRPeaks = []
q75GSRPeaks = []
yTruth = []

##########################################################################################

feature_dictP = {}
datasetP = []
meanSpeech = []
maxSpeech = []
minSpeech = []
stdSpeech = []
SpeechSlope = []
skewSpeech = []
kurtSpeech = []
yTruthP = []

##########################################################################################

def time_in_seconds(t):
	h, m, s = t.tm_hour, t.tm_min, t.tm_sec
	t_secs = s + 60 * (m + 60 * h)
	return t_secs

def get_slope(eda, time):
	slope = stats.linregress(time, eda)
	return slope

def get_spikes(eda):

	x = np.array(eda)
	t = argrelmax(x)

	spikes = x[t]
	sortedSpikes = np.sort(spikes)

	numPeak = len(spikes)
	meanPeak = np.nan_to_num(np.mean(spikes))

	if len(sortedSpikes) != 0:
		peak25 = np.percentile(sortedSpikes, 25)
		peak50 = np.percentile(sortedSpikes, 50)
		peak75 = np.percentile(sortedSpikes, 75)
	else:
		peak25 = 0
		peak50 = 0
		peak75 = 0

	meanGSRPeaks.append(meanPeak)
	numGSRPeaks.append(numPeak)
	q25GSRPeaks.append(peak25)
	q50GSRPeaks.append(peak50)
	q75GSRPeaks.append(peak75)	

##########################################################################################

def get_features_GSR(actualValues, sizeData):
	currEDA = []
	currTime = []

	currRow = actualValues[0]
	sampleTime = time.strptime(currRow[0], '%H:%M:%S.%f')
	sampleTime_secs = time_in_seconds(sampleTime)
	i = 0

	while i < sizeData:

		currRow = actualValues[i]
		t = time.strptime(currRow[0], '%H:%M:%S.%f')
		t_secs = time_in_seconds(t)

		if t_secs - sampleTime_secs <= 10:
			currEDA.append(currRow[6])
			currTime.append(t_secs)
			if t_secs - sampleTime_secs == 5:
				next_index = i
				temp_sample_time = t_secs
			i += 1
		else:
			currEDA = [float(x) for x in currEDA]
			# print(currEDA)
			meanGSR.append(np.mean(currEDA))
			minGSR.append(np.min(currEDA))
			maxGSR.append(np.max(currEDA))

			slopeVal = get_slope(currEDA, currTime)
			GSRSlopes.append(slopeVal.slope)
			# print(GSRSlopes)

			get_spikes(currEDA)

			currEDA = []
			currTime = []
			i = next_index
			sampleTime_secs = temp_sample_time

##########################################################################################

def rank_1_feature(flag):

	if flag == 0:
		feature_names = ['GSRSlopes', 'meanGSR', 'maxGSR', 'minGSR', 'q25GSRPeaks', 'q50GSRPeaks', 'q75GSRPeaks', 'meanGSRPeaks', 'numGSRPeaks']
		Y = np.array(yTruth)
	if flag == 1:
		feature_names = ['minSpeech', 'maxSpeech', 'meanSpeech', 'kurtSpeech', 'skewSpeech', 'stdSpeech', 'SpeechSlope']
		Y = np.array(yTruthP)

	f1Vals = []

	for i in range(0, len(feature_names)):

		X = np.array(feature_dict[feature_names[i]])

		sizeData = len(X)
		X = np.reshape(X, (sizeData, 1))

		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

		# clf = KNeighborsClassifier(n_neighbors=3)
		clf = RandomForestClassifier(max_depth=4)
		clf.fit(x_train, y_train)

		yPredicted = clf.predict(x_test)
		f1Score = f1_score(y_test, yPredicted)

		f1Vals.append(f1Score)

	print('F1 Scores: ')
	print(f1Vals)

	if flag == 0:
		x_axis = np.arange(9)
	if flag == 1:
		x_axis = np.arange(7)

	plt.bar(x_axis, f1Vals)
	plt.xticks(x_axis, feature_names)
	plt.show()

##########################################################################################

def rank_2_feature(flag):

	if flag == 0:
		feature_names = ['GSRSlopes', 'meanGSR', 'maxGSR', 'minGSR', 'q25GSRPeaks', 'q50GSRPeaks', 'q75GSRPeaks', 'meanGSRPeaks', 'numGSRPeaks']
		Y = np.array(yTruth)
	if flag == 1:
		feature_names = ['minSpeech', 'maxSpeech', 'meanSpeech', 'kurtSpeech', 'skewSpeech', 'stdSpeech', 'SpeechSlope']
		Y = np.array(yTruthP)

	f1Vals = []
	maxVal = 0.00000000000001

	for i in range(0, len(feature_names)):

		j = i + 1
		while j <= len(feature_names)-1:

			X1 = np.array(feature_dict[feature_names[i]])
			X2 = np.array(feature_dict[feature_names[j]])
			X = np.column_stack((X1, X2))

			x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

			# clf = KNeighborsClassifier(n_neighbors=3)
			clf = RandomForestClassifier(max_depth=4)
			clf.fit(x_train, y_train)

			yPredicted = clf.predict(x_test)
			f1Score = f1_score(y_test, yPredicted)

			if f1Score > maxVal:
				maxVal = f1Score
				best_i = i
				best_j = j

			f1Vals.append(f1Score)
			j += 1

	print(f1Vals)
	print('Best 2 features are:')
	print(feature_names[best_i])
	print(feature_names[best_j])

##########################################################################################

def forward_feature_Selection_GSR(numFeaturesFS):

	# if flag == 0:
	feature_names = ['GSRSlopes', 'meanGSR', 'maxGSR', 'minGSR', 'q25GSRPeaks', 'q75GSRPeaks', 'meanGSRPeaks', 'numGSRPeaks', 'meanSpeech', 'minSpeech', 'maxSpeech', 'SpeechSlope', 'kurtSpeech', 'skewSpeech', 'stdSpeech']

	f1Vals = []

	X = np.array(feature_dict['q50GSRPeaks'])
	Y1 = np.array(yTruth)
	Y2 = np.array(yTruthP)
	Y = np.maximum(Y1, Y2) 

	i = 2

	print('Best '+str(numFeaturesFS)+' features are:')

	while i <= numFeaturesFS:

		j = 0
		index = -1
		maxScore = 0.000000000000000000001

		while j < len(feature_names):

			X2 = np.array(feature_dict[feature_names[j]])
			XTemp = np.column_stack((X, X2))

			x_train, x_test, y_train, y_test = train_test_split(XTemp, Y, test_size=0.3)

			# clf = KNeighborsClassifier(n_neighbors=3)
			clf = RandomForestClassifier(max_depth=5)
			clf.fit(x_train, y_train)

			yPredicted = clf.predict(x_test)
			f1Score = f1_score(y_test, yPredicted)

			if f1Score > maxScore:
				index = j
				maxScore = f1Score

			j += 1

		if index != -1:
			print(feature_names[index])
			X2 = np.array(feature_dict[feature_names[index]])
			X = np.column_stack((X, X2))
			feature_names.remove(feature_names[index])

		i += 1

##########################################################################################

def get_features_Speech(actualValues, sizeData):
	currPitch = []
	currTime = []

	sampleTime_secs = 0
	i = 0
	prevIndex = i

	while i < sizeData:

		t_secs = i * 0.032

		if i - prevIndex <= 88:
			currPitch.append(actualValues[i][0])
			currTime.append(t_secs)
			if i - prevIndex == 44:
				nextIndex = i
				temp_sample_time = t_secs
			i += 1
		else:
			currPitch = [float(x) for x in currPitch]

			meanSpeech.append(np.mean(currPitch))
			minSpeech.append(np.min(currPitch))
			maxSpeech.append(np.max(currPitch))
			stdSpeech.append(np.std(currPitch))
			skewSpeech.append(stats.skew(currPitch))
			kurtSpeech.append(stats.kurtosis(currPitch))

			slopeVal = get_slope(currPitch, currTime)
			SpeechSlope.append(slopeVal.slope)

			currPitch = []
			currTime = []
			i = nextIndex
			prevIndex = i
			sampleTime_secs = temp_sample_time

##########################################################################################

def sensor_fusion_compare():


	X1 = np.array(feature_dict['skewSpeech'])
	X2 = np.array(feature_dict['maxSpeech'])
	X_S = np.column_stack((X1, X2))
	Y_S = np.array(yTruthP)

	X3 = np.array(feature_dict['maxGSR'])
	X4 = np.array(feature_dict['q25GSRPeaks'])
	X_G = np.column_stack((X3, X4))
	Y_G = np.array(yTruth)

	#best 2 GSR features
	x_train, x_test, y_train, y_test = train_test_split(X_G, Y_G, test_size=0.3)

	# clf = KNeighborsClassifier(n_neighbors=3)
	clf = RandomForestClassifier(max_depth=4)
	clf.fit(x_train, y_train)

	yPredicted = clf.predict(x_test)
	print('F1, accuracy, recall and precision values using only GSR features:')
	print(f1_score(y_test, yPredicted))
	print(accuracy_score(y_test, yPredicted))
	print(recall_score(y_test, yPredicted))
	print(average_precision_score(y_test, yPredicted))

	#best 2 Speech features
	x_train, x_test, y_train, y_test = train_test_split(X_S, Y_S, test_size=0.3)

	# clf = KNeighborsClassifier(n_neighbors=3)
	clf = RandomForestClassifier(max_depth=4)
	clf.fit(x_train, y_train)

	yPredicted = clf.predict(x_test)
	print('F1, accuracy, recall and precision values using only Speech features:')
	print(f1_score(y_test, yPredicted))
	print(accuracy_score(y_test, yPredicted))
	print(recall_score(y_test, yPredicted))
	print(average_precision_score(y_test, yPredicted))

	#best both GSR and Speech features
	X = np.column_stack((X_S, X_G))
	Y = np.maximum(Y_S, Y_G)

	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

	# clf = KNeighborsClassifier(n_neighbors=3)
	clf = RandomForestClassifier(max_depth=4)
	clf.fit(x_train, y_train)

	yPredicted = clf.predict(x_test)
	print('F1, accuracy, recall and precision values using both GSR and Speech features:')
	print(f1_score(y_test, yPredicted))
	print(accuracy_score(y_test, yPredicted))
	print(recall_score(y_test, yPredicted))
	print(average_precision_score(y_test, yPredicted))


##########################################################################################

def get_all_participants_GSRdata():
	prefix = 'sub'
	suffix = '.csv'

	for i in range(1, 11):

		tempDataset = []
		f = open(prefix+str(i)+suffix)
		fReader = csv.reader(f)

		for row in fReader:
			tempDataset.append(row)

		dataNoLabels = tempDataset[8:]
		dataset.append(dataNoLabels)

def get_y_GSR_labels():

	fy = open('y.csv')
	fReader = csv.reader(fy)

	for row in fReader:
		yTruth.append(int(row[0]))

def get_all_participants_Speechdata():
	prefix = 'subI'
	suffix = '.csv'

	for i in range(1, 9):

		tempDataset = []
		f = open(prefix+str(i)+suffix)
		fReader = csv.reader(f)

		for row in fReader:
			tempDataset.append(row)

		datasetP.append(tempDataset)

def get_y_Speech_labels():

	fy = open('yPitch.csv')
	fReader = csv.reader(fy)

	for row in fReader:
		yTruthP.append(int(row[0]))

##########################################################################################

if __name__ == '__main__':

	print('*************************************************************************')
	print('Extracting all participants data and ground truth labels for GSR and Speech data:')
	get_all_participants_GSRdata()
	get_all_participants_Speechdata()
	get_y_GSR_labels()
	get_y_Speech_labels()

	print('Extracting Features for GSR data:')
	for i in range(0, 10):
		actualValues = dataset[i]
		sizeData = len(actualValues)

		# for j in range(0, sizeData):
		# 	print(actualValues[j])

		get_features_GSR(actualValues, sizeData)

	#adding GSR features to a dictionary file for easy access
	feature_dict["GSRSlopes"] = GSRSlopes
	feature_dict["meanGSR"] = meanGSR
	feature_dict["minGSR"] = minGSR
	feature_dict["maxGSR"] = maxGSR
	feature_dict["numGSRPeaks"] = numGSRPeaks
	feature_dict["meanGSRPeaks"] = meanGSRPeaks
	feature_dict["q25GSRPeaks"] = q25GSRPeaks
	feature_dict["q50GSRPeaks"] = q50GSRPeaks
	feature_dict["q75GSRPeaks"] = q75GSRPeaks

	flag = 0
	print('*************************************************************************')
	print('Getting best feature of GSR data for stress classification:')
	rank_1_feature(flag)
	print('*************************************************************************')
	print('Getting 2 best features of GSR data for stress classification')
	rank_2_feature(flag)

	yTruthP = yTruthP[:-16]
	print('Extracting Features for Speech data:')
	for i in range(0, 8):
		actualValues = datasetP[i]
		sizeData = len(actualValues)

		get_features_Speech(actualValues, sizeData)

	#adding Speech features to a dictionary file
	feature_dict["meanSpeech"] = meanSpeech[:-16]
	feature_dict["minSpeech"] = minSpeech[:-16]
	feature_dict["maxSpeech"] = maxSpeech[:-16]
	feature_dict["SpeechSlope"] = SpeechSlope[:-16]
	feature_dict["skewSpeech"] = skewSpeech[:-16]
	feature_dict["kurtSpeech"] = kurtSpeech[:-16]
	feature_dict["stdSpeech"] = stdSpeech[:-16]

	flag = 1
	print('*************************************************************************')
	print('Getting best feature of Speech data for stress classification:')
	rank_1_feature(flag)
	print('*************************************************************************')
	print('Getting 2 best features of Speech data for stress classification')
	rank_2_feature(flag)

	numFeaturesFS = 6
	print('*************************************************************************')
	print('Doing forward feature selection for stress classification')
	forward_feature_Selection_GSR(numFeaturesFS)

	print('*************************************************************************')
	print('Comparing sensor fusion for stress classification')
	sensor_fusion_compare()


	




	


