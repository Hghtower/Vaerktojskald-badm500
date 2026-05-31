import matplotlib.pyplot as plt
import pandas as pd


# Script for creating the plot of the performance of the models trained on dataset_V2 vs dataset_V3

# Dataset V2
trainlossv2 = pd.read_json('graph_data/finalruns/datasetv2/trainloss.json')
trainaccv2 = pd.read_json('graph_data/finalruns/datasetv2/trainacc.json')

evallossv2 = pd.read_json('graph_data/finalruns/datasetv2/evalloss.json')
evalaccv2 = pd.read_json('graph_data/finalruns/datasetv2/evalacc.json')

# Dataset V3
trainlossv3 = pd.read_json('graph_data/finalruns/datasetv3/trainloss.json')
trainaccv3 = pd.read_json('graph_data/finalruns/datasetv3/trainacc.json')

evallossv3 = pd.read_json('graph_data/finalruns/datasetv3/evalloss.json')
evalaccv3 = pd.read_json('graph_data/finalruns/datasetv3/evalacc.json')

plt.subplot(2,2,1)
plt.plot(trainlossv2[1], trainlossv2[2], label='Dataset V2')
plt.plot(trainlossv3[1], trainlossv3[2], label='Dataset V3')
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('Training Loss')
plt.grid(True)
plt.legend()

plt.subplot(2,2,2)
plt.plot(trainaccv2[1], trainaccv2[2], label='Dataset V2')
plt.plot(trainaccv3[1], trainaccv3[2], label='Dataset V3')
plt.xlabel('steps')
plt.ylabel('mean token accuracy')
plt.title('Training Accuracy')
plt.grid(True)
plt.legend()

plt.subplot(2,2,3)
plt.plot(evallossv2[1], evallossv2[2], label='Dataset V2')
plt.plot(evallossv3[1], evallossv3[2], label='Dataset V3')
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('Eval Loss')
plt.grid(True)
plt.legend()

plt.subplot(2,2,4)
plt.plot(evalaccv2[1], evalaccv2[2], label='Dataset V2')
plt.plot(evalaccv3[1], evalaccv3[2], label='Dataset V3')
plt.xlabel('steps')
plt.ylabel('mean token accuracy')
plt.title('Eval Accuracy')
plt.grid(True)
plt.legend()

plt.show()