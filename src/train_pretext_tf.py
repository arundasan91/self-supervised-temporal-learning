import os
import time
import pickle
import random
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Reshape, Permute
from tensorflow.keras.layers import Conv1D, Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.constraints import max_norm, unit_norm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

fname_tag = 'all_sessions_942_970_971_'

with open('../dataset/pretext/pretext_dataset_'+ fname_tag +'x.npy','rb') as f: X_array = np.load(f)
with open('../dataset/pretext/pretext_dataset_'+ fname_tag +'ywin.npy','rb') as f: Y_array_window = np.load(f)
with open('../dataset/pretext/pretext_dataset_'+ fname_tag +'ytask.npy','rb') as f: Y_array_task = np.load(f)

num_windows = np.unique(Y_array_window).__len__()
num_tasks = np.unique(Y_array_task).__len__()

print("X_array.shape: ", X_array.shape)


Y_array_window = to_categorical(np.ravel(Y_array_window), num_classes=num_windows)
Y_array_task = to_categorical(np.ravel(Y_array_task), num_classes=num_tasks)

def define_multihead_model(dropout=0.25, num_windows=1, num_tasks=1):
    K.set_image_data_format('channels_first')

    clear_session()

    channels = 17
    timesteps = 87 # upsampled 58 fps

    inputs = Input(shape=(channels, timesteps))

    input_permute = Permute((1, 2), input_shape=(channels, timesteps))(inputs)
    input_reshape = Reshape((1, channels, timesteps))(input_permute)

    conv2d_1 = Conv2D(16, (1,channels), activation='linear', input_shape=(channels, timesteps), padding='same')(input_reshape)
    conv2d_1_bn = BatchNormalization()(conv2d_1)

    conv2d_2DW = DepthwiseConv2D((channels,1), use_bias=False, activation='linear', depth_multiplier=2, padding='valid', kernel_constraint=max_norm(1.))(conv2d_1_bn)
    conv2d_2DW_bn = BatchNormalization()(conv2d_2DW)
    conv2d_2DW_bn_act = Activation('elu')(conv2d_2DW_bn)

    conv2d_2DW_bn_act_avpool = AveragePooling2D((1,4))(conv2d_2DW_bn_act)
    conv2d_2DW_bn_act_avpool_dp = Dropout(rate=dropout)(conv2d_2DW_bn_act_avpool)

    conv2d_3Sep = SeparableConv2D(32, (1, channels-1), activation='linear', padding='same')(conv2d_2DW_bn_act_avpool_dp)
    conv2d_3Sep_bn = BatchNormalization()(conv2d_3Sep)
    conv2d_3Sep_bn_act = Activation('elu')(conv2d_3Sep_bn)

    conv2d_3Sep_bn_act_avgpool = AveragePooling2D((1,8))(conv2d_3Sep_bn_act)
    conv2d_3Sep_bn_act_avgpool_dp = Dropout(rate=dropout)(conv2d_3Sep_bn_act_avgpool)

    flatten_1 = Flatten()(conv2d_3Sep_bn_act_avgpool_dp)
    dense_1 = Dense(64, activation='elu', 
                    # kernel_constraint=max_norm(0.25),
                    name='embedding')(flatten_1)

    predictions_window = Dense(num_windows, activation='softmax', 
                        # kernel_constraint=max_norm(0.25), 
                        name='predictions_window')(dense_1)
    
    predictions_task = Dense(num_tasks, activation='softmax', 
                        # kernel_constraint=max_norm(0.25), 
                        name='predictions_task')(dense_1)
    
    model = Model(inputs=inputs, outputs=(predictions_window, predictions_task))
    
    return model
    
def plot_history(history, acc_tag='accuracy', loss_tag='loss'):
    # Plot training & validation accuracy values
    plt.plot(history.history[acc_tag])
    plt.plot(history.history['val_'+acc_tag])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history[loss_tag])
    plt.plot(history.history['val_'+loss_tag])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_auc(fpr, tpr, auc_value):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC-ROC (area = {:.3f})'.format(auc_value))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    plt.show()

start_time = time.time()
# fix random seed for reproducibility
seed = 13
np.random.seed(seed)

X_array_, X_test, Y_array_window_, y_test_window, Y_array_task_, y_test_task = train_test_split(X_array, Y_array_window, Y_array_task, test_size=0.2, random_state=seed)

kfold = KFold(n_splits=5, random_state=seed)

accuracies_window = []
precisions_window = []
recalls_window = []
f1s_window = []
aucs_window = []

accuracies_task = []
precisions_task = []
recalls_task = []
f1s_task = []
aucs_task = []

count = 1

for train, val in kfold.split(X_array_):
    print("Processing Fold ", count)
    
    print("Train Data Statistics")
    print("X_array.shape: ", X_array_[train].shape)
    
    print("Validation Data Statistics")
    print("X_array.shape: ", X_array_[val].shape)
    
    print("Test Data Statistics")
    print("X_array.shape: ", X_test.shape)

#     X_train = tf.data.Dataset.from_tensor_slices(X_array_[train])
#     Y_train_window = tf.data.Dataset.from_tensor_slices(Y_array_window[train])
#     Y_train_task = tf.data.Dataset.from_tensor_slices(Y_array_task[train])
    
#     X_val = tf.data.Dataset.from_tensor_slices(X_array_[val])
#     Y_val_window = tf.data.Dataset.from_tensor_slices(Y_array_window[val])
#     Y_val_task = tf.data.Dataset.from_tensor_slices(Y_array_task[val])
    
    count += 1
    try:
        del(model)
    except NameError:
        pass

    hyper_parameters = {}
    hyper_parameters['lr']=0.01
    hyper_parameters['momentum']=0.9
    hyper_parameters['lr_factor'] = 0.5
    hyper_parameters['lr_patience'] = 50
    hyper_parameters['lr_cooldown'] = 50
    hyper_parameters['min_lr'] = 0.0001
    hyper_parameters['early_min_delta'] = 0.001
    hyper_parameters['early_patience'] = 5
    hyper_parameters['batch_size'] = 256
    hyper_parameters['epochs'] = 50
    hyper_parameters['dropout']=0.50
    hyper_parameters['loss'] = {
                                "predictions_window": "categorical_crossentropy",
                                "predictions_task": "categorical_crossentropy",
                               }
    hyper_parameters['lossWeights'] = {"predictions_window": 1.0, "predictions_task": 1.0}
    
    sgd = optimizers.SGD(lr=hyper_parameters['lr'], momentum=hyper_parameters['momentum'], nesterov=False)
    adam = optimizers.Adam(learning_rate=hyper_parameters['lr'], amsgrad=False)
    hyper_parameters['optimizer'] = adam
    
    SHUFFLE_BUFFER_SIZE = 100

#     train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(hyper_parameters['batch_size'])
#     validation_dataset = validation_dataset.batch(hyper_parameters['batch_size'])
    
    model = define_multihead_model(num_windows=num_windows, num_tasks=num_tasks)
    
    reduce_lr = ReduceLROnPlateau(monitor='loss', 
                                  factor=hyper_parameters['lr_factor'], 
                                  patience=hyper_parameters['lr_patience'], 
                                  cooldown=hyper_parameters['lr_cooldown'], 
                                  min_lr=hyper_parameters['min_lr'])
    
    early_stop = EarlyStopping(monitor='val_loss', 
                               min_delta=hyper_parameters['early_min_delta'], 
                               patience=hyper_parameters['early_patience'])
    
    log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if type(hyper_parameters['loss']) == dict:
        model.compile(loss=hyper_parameters['loss'],
                      loss_weights=hyper_parameters['lossWeights'],
                      optimizer=hyper_parameters['optimizer'], 
                      metrics=['accuracy'])
    else:
        model.compile(loss=hyper_parameters['loss'],
                      optimizer=hyper_parameters['optimizer'], 
                      metrics=['accuracy'])

    history = model.fit(x=X_array_[train], 
                        y={"predictions_window": Y_array_window_[train],
                          "predictions_task": Y_array_task_[train]},
                        validation_data=(X_array_[val], 
                         {"predictions_window": Y_array_window_[val],
                          "predictions_task": Y_array_task_[val]}),
                        epochs=hyper_parameters['epochs'], 
                        batch_size=hyper_parameters['batch_size'], 
                        verbose=1 if count==1 else 0,
                        shuffle=True,
                        callbacks=[reduce_lr, early_stop])#, tensorboard_callback])
    
    plot_history_curves = False
    if plot_history_curves:
        plot_history(history, acc_tag='predictions_window_accuracy', loss_tag='predictions_window_loss')
        plot_history(history, acc_tag='predictions_task_accuracy', loss_tag='predictions_task_loss')


    test_output = model.evaluate(X_test, {"predictions_window": y_test_window, "predictions_task": y_test_task}, verbose=0, return_dict=True)
    
    y_pred = model.predict(X_test)
    y_pred_window_ = np.argmax(y_pred[0], axis=1)
    y_pred_task_ = np.argmax(y_pred[1], axis=1)

    y_test_window_ = np.argmax(y_test_window, axis=1)
    y_test_task_ = np.argmax(y_test_task, axis=1)

    precision_window = precision_score(y_test_window_, y_pred_window_, average='macro')
    recall_window = recall_score(y_test_window_, y_pred_window_, average='macro')
    f1_window = f1_score(y_test_window_, y_pred_window_, average='macro')

    precision_task = precision_score(y_test_task_, y_pred_task_, average='macro')
    recall_task = recall_score(y_test_task_, y_pred_task_, average='macro')
    f1_task = f1_score(y_test_task_, y_pred_task_, average='macro')

    print("Accuracy: {:.2f}%, Recall: {:.3f}, Precision: {:.3f}, F1: {:.3f}".format(test_output['predictions_window_accuracy']*100, recall_window, precision_window, f1_window))
    print("Accuracy: {:.2f}%, Recall: {:.3f}, Precision: {:.3f}, F1: {:.3f}".format(test_output['predictions_task_accuracy']*100, recall_task, precision_task, f1_task))

    accuracies_window.append(test_output['predictions_window_accuracy']*100)
    precisions_window.append(precision_window)
    recalls_window.append(recall_window)
    f1s_window.append(f1_window)
    
    accuracies_task.append(test_output['predictions_task_accuracy']*100)
    precisions_task.append(precision_task)
    recalls_task.append(recall_task)
    f1s_task.append(f1_task)
    
    if test_output['predictions_task_accuracy'] > 0.85 and recall_task > 0.85:
        model_name = '_pretext_one_session_'
        model_path = '../trained_models/' + str(datetime.date.today()) + model_name + '{:.3f}'.format(test_output['predictions_task_accuracy'])[-3:]
        print("Saving to: ", model_path + '.h5')
        model.save(model_path + '.h5')

        with open(model_path + '_history.pkl','wb') as f: pickle.dump(model.history.history, f)
        with open(model_path + '_params.pkl','wb') as f: pickle.dump(model.history.params, f)
        with open(model_path + '_hyperparameters.pkl','wb') as f: pickle.dump(hyper_parameters, f)

    if count==2:
        print("Run Time for Fold ", count-1, ":", time.time()-start_time)
    print("########=================########")
    
print("Run Time: ", time.time()-start_time)

print("Training Statistics")
metrics_df_window = pd.DataFrame.from_dict({'Accuracy': accuracies_window, 'F1': f1s_window, 'Recall': recalls_window, 'Precision': precisions_window})
metrics_df_task = pd.DataFrame.from_dict({'Accuracy': accuracies_task, 'F1': f1s_task, 'Recall': recalls_task, 'Precision': precisions_task})

print(((metrics_df_task+metrics_df_window)/2).describe())
