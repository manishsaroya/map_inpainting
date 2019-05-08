import numpy as np
import csv
from numpy import concatenate

content = []
with open('experiments_all_cumulative_score_32.csv') as file:
	readCSV = csv.reader(file, delimiter=',')
	print(readCSV)
	for row in readCSV:
		content.append(list(int(eval(row[i])) for i in range(5, len(row))))

content = np.transpose(np.array(content))  #row contain the score from different maps

mean_list = [[],[],[],[],[]]
std_list = [[],[],[],[],[]]
for i in content:
	# segregate all the frontier values 
	values = [[],[],[],[],[]]
	for j in range(len(i)):
		values[j%5].append(i[j])
	
	print(values)
	#Compute mean and std_dev
	for v in range(len(values)):
		mean_list[v].append(np.mean(values[v]))
		std_list[v].append(np.std(values[v]))

explore_type = ['Value', 'Quarter', 'Closest', 'Square_root','Normal']
with open('formatted_cumulative_score_32.csv', mode='w') as file:
	format_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for x in range(len(mean_list)):
		to_write = concatenate([[explore_type[x]],['Mean'], mean_list[x]])
		format_writer.writerow(to_write)
		to_write = concatenate([[explore_type[x]],['Stdev'], std_list[x]])
		format_writer.writerow(to_write)


print(content)
print(len(content))
