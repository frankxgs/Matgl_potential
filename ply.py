import matplotlib.pyplot as plt
import pandas as pd


metrics = pd.read_csv('logs/RP_training/version_0/metrics.csv')
epochs = metrics['epoch'].dropna()
plt.plot(epochs, metrics['train_MAE'].dropna(), label='Train MAE')
plt.plot(epochs, metrics['val_MAE'].dropna(), label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training and Validation MAE over Epochs')
_ = plt.legend()
plt.show()