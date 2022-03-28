from tensorflow.keras.callbacks import Callback
import numpy as np

from models.losses import acc_trim

class Metrics(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs=None):
        self.val_accs = []

    def on_epoch_end(self, epoch, logs=None):
        # For each validation batch data
        for batch_index in range(0, len(self.validation_data)):
            # Get batch target values
            temp_targ = self.validation_data[batch_index][1].squeeze()
            temp_x = self.validation_data[batch_index][0]
            # Get batch prediction values
            temp_pred = (np.asarray(self.model.predict(temp_x))).round().squeeze()
            # Append them to the corresponding output objects
            if batch_index == 0:
                val_targ = temp_targ
                val_pred = temp_pred
            else:
                val_targ = np.vstack((val_targ, temp_targ))
                val_pred = np.vstack((val_pred, temp_pred))

        val_acc = round(acc_trim(val_targ, val_pred).numpy(), 4)

        self.val_accs.append(val_acc)
        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        logs['val_acc'] = val_acc
        print(f'Epoch {epoch}: Val Acc: {val_acc}')
