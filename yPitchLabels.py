import numpy as np
import csv
import time
from scipy import stats
from scipy.signal import argrelmax

dataset = []
fy = open('yPitch.csv', 'a')
writer = csv.writer(fy)

##########################################################################################

def get_features_Speech(actualValues, sizeData):

	sampleTime_secs = 0
	i = 0
	prevIndex = 0

	while i < sizeData:

		t_secs = i * 0.032

		if i - prevIndex <= 88:
			if i - prevIndex == 44:
				nextIndex = i
				temp_sample_time = t_secs
			i += 1
		else:

			writer.writerow('1')

			i = nextIndex
			prevIndex = i
			sampleTime_secs = temp_sample_time

def get_pitch_data():
	prefix = 'subI'
	suffix = '.csv'

	for i in range(1, 9):

		tempDataset = []
		f = open(prefix+str(i)+suffix)
		fReader = csv.reader(f)

		for row in fReader:
			tempDataset.append(row)

		dataset.append(tempDataset)

##########################################################################################

if __name__ == '__main__':

	get_pitch_data()

	actualValues = dataset[7]
	sizeData = len(actualValues)


	get_features_Speech(actualValues, sizeData)

	fy.close()

