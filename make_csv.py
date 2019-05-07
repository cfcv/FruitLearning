from sklearn.datasets import load_files

data = load_files("../FruitLearning/Resources/database")
size = [0, 0, 0, 0]
f = open('targets.csv', 'w')
for filename in data['filenames']:
    if(filename[36:38] == 'su'):
        f.write(filename + " 1\n")
        #print(filename)
    elif(filename[36] == 'a'):
        f.write(filename + " 2\n")
        #print(filename)
    elif(filename[36] == 's'):
        #print(filename)
        f.write(filename + " 0\n")
    else:
        #print(filename)
        f.write(filename + " 3\n")

print(len(data['filenames']))
f.close()