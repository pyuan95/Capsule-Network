import sys
import matplotlib.pyplot as plt

x = open(sys.argv[1],'r')
data = x.read()
data = data.split("\n")
data = [line.split(" ") for line in data]
data_filtered = [line for line in data if len(line) > 2 and line[2] == "Loss"]
loss = []
accuracy = []
x = []
for index, line in enumerate(data_filtered):
    x.append(index + 1)
    loss.append(float(line[3]))
    accuracy.append(float(line[-1]))

plt.plot(x, loss, label = "loss")
plt.plot(x, accuracy, label = "accuracy")
plt.legend()
plt.show()