import matplotlib.pyplot as plt

# 文件中已有的 TrainLoss
train_loss = [
    1.1889, 0.9984, 0.9609, 0.9418, 0.9293,
    0.9200, 0.9128, 0.9071, 0.9025, 0.8986,
    0.8952, 0.8922, 0.8897, 0.8874, 0.8853
]

# 你修改后的 ValLoss
val_loss = [
    1.2517, 1.2305, 1.2201, 1.2158, 1.2132,
    1.2120, 1.2115, 1.2112, 1.2110, 1.2109,
    1.2108, 1.2107, 1.2107, 1.2106, 1.2106
]

epochs = range(1, 16)

plt.plot(epochs, train_loss, label='TrainLoss')
plt.plot(epochs, val_loss, label='ValLoss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM TrainLoss & ValLoss')
plt.legend()
plt.grid(True)
plt.show()

