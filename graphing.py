import matplotlib.pyplot as plt
import pandas as pd

#Warm up steps
# Accuracy
data1 = pd.read_json('graph_data/0WarmupAcc.json')
data2 = pd.read_json('graph_data/5WarmupAcc.json')
data3 = pd.read_json('graph_data/10WarmupAcc.json')
# Loss  
data4 = pd.read_json('graph_data/0WarmupLoss.json')
data5 = pd.read_json('graph_data/5WarmupLoss.json')
data6 = pd.read_json('graph_data/10WarmupLoss.json')

plt.subplot(1, 2, 1)
plt.plot(data1[1], data1[2], label='0 steps')
plt.plot(data2[1], data2[2], label='5 steps')
plt.plot(data3[1], data3[2], label='10 steps')
plt.grid(True)
plt.xlabel('steps')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(data4[1], data4[2], label='0 steps')
plt.plot(data5[1], data5[2], label='5 steps')
plt.plot(data6[1], data6[2], label='10 steps')
plt.grid(True)
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()



