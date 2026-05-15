import matplotlib.pyplot as plt
import pandas as pd


def warmup_graph():
    #Warm up steps
    # Accuracy
    data1 = pd.read_json('graph_data/warmup/0WarmupAcc.json')
    data2 = pd.read_json('graph_data/warmup/5WarmupAcc.json')
    data3 = pd.read_json('graph_data/warmup/10WarmupAcc.json')
    # Loss  
    data4 = pd.read_json('graph_data/warmup/0WarmupLoss.json')
    data5 = pd.read_json('graph_data/warmup/5WarmupLoss.json')
    data6 = pd.read_json('graph_data/warmup/10WarmupLoss.json')

    plt.subplot(1, 2, 1)
    plt.plot(data1[1], data1[2], label='0 steps')
    plt.plot(data2[1], data2[2], label='5 steps')
    plt.plot(data3[1], data3[2], label='10 steps')
    plt.grid(True)
    plt.xlabel('steps')
    plt.ylabel('mean token accuracy')
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

    #plt.tight_layout()
    plt.show()

def gradAccuGraph():
    # Accuracy
    data1 = pd.read_json('graph_data/GradAccu/grad1acc.json')
    data2 = pd.read_json('graph_data/learningRate/1e-4acc.json')
    data3 = pd.read_json('graph_data/GradAccu/grad6acc.json')
    data7 = pd.read_json('graph_data/GradAccu/grad8acc.json')
    data9 = pd.read_json('graph_data/GradAccu/grad10acc.json')
    # Loss  
    data4 = pd.read_json('graph_data/GradAccu/grad1loss.json')
    data5 = pd.read_json('graph_data/learningRate/1e-4loss.json')
    data6 = pd.read_json('graph_data/GradAccu/grad6loss.json')
    data8 = pd.read_json('graph_data/GradAccu/grad8loss.json')
    data10 = pd.read_json('graph_data/GradAccu/grad10loss.json')

    plt.subplot(1, 2, 1)
    plt.plot(data1[1], data1[2], label='1')
    plt.plot(data2[1], data2[2], label='4')
    plt.plot(data3[1], data3[2], label='6')
    plt.plot(data7[1], data7[2], label='8')
    plt.plot(data9[1], data9[2], label='10')
    plt.grid(True)
    plt.xlabel('steps')
    plt.ylabel('mean token accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(data4[1], data4[2], label='1')
    plt.plot(data5[1], data5[2], label='4')
    plt.plot(data6[1], data6[2], label='6')
    plt.plot(data8[1], data8[2], label='8')
    plt.plot(data10[1], data10[2], label='10')
    plt.grid(True)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()

    #plt.tight_layout()
    plt.show()

def learnRateGraph():
    # Accuracy
    data1 = pd.read_json('graph_data/learningRate/1e-4acc.json')
    data2 = pd.read_json('graph_data/learningRate/2e-3acc.json')
    data3 = pd.read_json('graph_data/learningRate/2e-5acc.json')
    data4 = pd.read_json('graph_data/learningRate/2e-6acc.json')
    data5 = pd.read_json('graph_data/learningRate/3e-4acc.json')
    data6 = pd.read_json('graph_data/learningRate/4e-4acc.json')
    data7 = pd.read_json('graph_data/learningRate/5e-5acc.json')
    data8 = pd.read_json('graph_data/warmup/0WarmupAcc.json')

    # Loss  
    data9 = pd.read_json('graph_data/learningRate/1e-4loss.json')
    data10 = pd.read_json('graph_data/learningRate/2e-3loss.json')
    data11 = pd.read_json('graph_data/learningRate/2e-5loss.json')
    data12 = pd.read_json('graph_data/learningRate/2e-6loss.json')
    data13 = pd.read_json('graph_data/learningRate/3e-4loss.json')
    data14 = pd.read_json('graph_data/learningRate/4e-4loss.json')
    data15 = pd.read_json('graph_data/learningRate/5e-5loss.json')
    data16 = pd.read_json('graph_data/warmup/0WarmupLoss.json')

    plt.subplot(1, 2, 1)

    plt.plot(data1[1], data1[2], label='1e-4')
    plt.plot(data2[1], data2[2], label='2e-3')
    plt.plot(data3[1], data3[2], label='2e-5')
    plt.plot(data4[1], data4[2], label='2e-6')
    plt.plot(data5[1], data5[2], label='3e-4')
    plt.plot(data6[1], data6[2], label='4e-4')
    plt.plot(data7[1], data7[2], label='5e-5')
    plt.plot(data8[1], data8[2], label='2e-4')

    plt.grid(True)
    plt.xlabel('steps')
    plt.ylabel('mean token accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(data9[1], data9[2], label='1e-4')
    plt.plot(data10[1], data10[2], label='2e-3')
    plt.plot(data11[1], data11[2], label='2e-5')
    plt.plot(data12[1], data12[2], label='2e-6')
    plt.plot(data13[1], data13[2], label='3e-4')
    plt.plot(data14[1], data14[2], label='4e-4')
    plt.plot(data15[1], data15[2], label='5e-5')
    plt.plot(data16[1], data16[2], label='2e-4')

    plt.grid(True)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()

    #plt.tight_layout()
    plt.show()

   

def epochGraph():
    # Accuracy
    data1 = pd.read_json('graph_data/GradAccu/grad8acc.json')
    data2 = pd.read_json('graph_data/epochs/epoch3acc.json')
    data3 = pd.read_json('graph_data/epochs/epoch4acc.json')
    data7 = pd.read_json('graph_data/epochs/epoch5acc.json')

    # Loss  
    data4 = pd.read_json('graph_data/GradAccu/grad8loss.json')
    data5 = pd.read_json('graph_data/epochs/epoch3loss.json')
    data6 = pd.read_json('graph_data/epochs/epoch4loss.json')
    data8 = pd.read_json('graph_data/epochs/epoch5loss.json')

    plt.subplot(1, 2, 1)

    plt.plot(data1[1], data1[2], label='2 epochs')
    plt.plot(data2[1], data2[2], label='3 epochs')
    plt.plot(data3[1], data3[2], label='4 epochs')
    plt.plot(data7[1], data7[2], label='5 epochs')

    plt.grid(True)
    plt.xlabel('steps')
    plt.ylabel('mean token accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(data4[1], data4[2], label='2 epochs')
    plt.plot(data5[1], data5[2], label='3 epochs')
    plt.plot(data6[1], data6[2], label='4 epochs')
    plt.plot(data8[1], data8[2], label='5 epochs')

    plt.grid(True)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()

    #plt.tight_layout()
    plt.show()


 

if __name__ == "__main__":
    warmup_graph()
    gradAccuGraph()
    learnRateGraph()
    epochGraph()
