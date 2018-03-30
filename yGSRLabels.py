import numpy as np
import csv
import time
from scipy import stats
from scipy.signal import argrelmax

Y = []
dataset = []
fy = open('y.csv', 'a')
writer = csv.writer(fy)

##########################################################################################

def time_in_seconds(t):
	h, m, s = t.tm_hour, t.tm_min, t.tm_sec
	t_secs = s + 60 * (m + 60 * h)
	return t_secs

def get_features_GSR(actualValues, sizeData):

	currRow = actualValues[0]
	sampleTime = time.strptime(currRow[0], '%H:%M:%S.%f')
	sampleTime_secs = time_in_seconds(sampleTime)
	i = 0

	while i < sizeData:

		currRow = actualValues[i]
		t = time.strptime(currRow[0], '%H:%M:%S.%f')
		t_secs = time_in_seconds(t)

		if t_secs - sampleTime_secs <= 10:
			if t_secs - sampleTime_secs == 5:
				next_index = i
				temp_sample_time = t_secs
			i += 1
		else:

			# if (t_secs >= 41090 and t_secs <= 41840) or (sampleTime_secs >= 41090 and sampleTime_secs <= 41840):
			# 	writer.writerow('1')
			# else:
			writer.writerow('0')

			i = next_index
			sampleTime_secs = temp_sample_time

def get_all_participants_data():
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

##########################################################################################

if __name__ == '__main__':

	get_all_participants_data()

	actualValues = dataset[9]
	sizeData = len(actualValues)


	get_features_GSR(actualValues, sizeData)

	fy.close()
