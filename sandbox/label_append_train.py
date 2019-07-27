import sys 

dataset = sys.argv[1]
print(dataset)

path = "data/{}/".format(dataset)
trn_X_file = open('{}/trn_X_Xf.txt'.format(path), 'r')
trn_Y_file = open('{}/trn_X_Y.txt'.format(path), 'r')
fout = open('{}/trn_X_XY.txt'.format(path), 'w')

num_trn, num_dim = map(int, trn_X_file.readline().split())
_, num_lab = map(int, trn_Y_file.readline().split())

offset = num_dim

fout.write(str(num_trn)+' '+str(num_dim+num_lab)+'\n')

lines_X = trn_X_file.readlines()
lines_Y = trn_Y_file.readlines()

for lineX, lineY in zip(lines_X, lines_Y):
	finalstr = lineX.strip()

	for each in lineY.split():
		lab = int(each.split(':')[0])
		finalstr += ' ' + str(offset + lab) + ':1'

	fout.write(finalstr+'\n')
