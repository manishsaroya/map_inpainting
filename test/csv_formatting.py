import numpy as np
import csv
from numpy import concatenate
import pdb
import copy
content = []
elements = 5

##### Order same as experiment.py #################
#explore_type = ['Value', 'Quarter', 'Closest', 'Square_root','Normal']
# keep the same order as experiment.py
#explore_type = ['Quater','Closest','Normal']
#explore_type = ['Quarter']
#explore_type = ['Normal', 'Square_root']
explore_type = ['Quarter', 'Closest', 'Square_root', 'Normal', 'Value']

with open('alpha.csv') as file:
	readCSV = csv.reader(file, delimiter=',')
	print(readCSV)
	for row in readCSV:
		content.append(list(eval(row[i]) for i in range(5, len(row))))  # remove int over eval, no idea why it was present in first place.

content = np.transpose(np.array(content))  #row contain the score from different maps

mean_list = []
std_list = []
for d in range(elements):
	mean_list.append([])
	std_list.append([])

for i in content:
	# segregate all the frontier values
	values = []
	for d in range(elements):
		values.append([])

	for j in range(len(i)):
		values[j%elements].append(i[j])
	divisor = copy.deepcopy(values[0])      # IMPORTANT PARAMETER Values[0] corresponds to Quater -> ground truth
	for k in range(elements):
		try:
			values[k] = (np.array(values[k]) / divisor) *100
		except ValueError:
			pdb.set_trace()
	
	#Compute mean and std_dev
	for v in range(len(values)):
		mean_list[v].append(np.mean(values[v]))
		std_list[v].append(np.std(values[v]))


with open('formatted_alpha.csv', mode='w') as file:
	format_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for x in range(len(mean_list)):
		to_write = concatenate([[explore_type[x]],['Mean'], mean_list[x]])
		format_writer.writerow(to_write)
		to_write = concatenate([[explore_type[x]],['Stdev'], std_list[x]])
		format_writer.writerow(to_write)


print(content)
print(len(content))
