import pickle
import random
import datetime
import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Reshape, Permute
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

paradigm = "all"

if paradigm == "all":
    # All Paradigms
    print("All Paradigms")
    with open(
        "../dataset/pickled_datasets/X_array_balanced_upsampled_S1S2_all_subjects.pkl",
        "rb",
    ) as f:
        X_array = pickle.load(f)
    with open(
        "../dataset/pickled_datasets/Y_array_balanced_upsampled_S1S2_all_subjects.pkl",
        "rb",
    ) as f:
        Y_array = pickle.load(f)

elif paradigm == "cw":
    # CueWord
    print("Cue-Word Paradigm")
    with open(
        "../dataset/pickled_datasets/X_array_balanced_upsampled_S1S2_CueWord_all_subjects.pkl",
        "rb",
    ) as f:
        X_array = pickle.load(f)
    with open(
        "../dataset/pickled_datasets/Y_array_balanced_upsampled_S1S2_CueWord_all_subjects.pkl",
        "rb",
    ) as f:
        Y_array = pickle.load(f)

elif paradigm == "wg":
    # WordGo
    print("Word-Go Paradigm")
    with open(
        "../dataset/pickled_datasets/X_array_balanced_upsampled_S1S2_WordGo_all_subjects.pkl",
        "rb",
    ) as f:
        X_array = pickle.load(f)
    with open(
        "../dataset/pickled_datasets/Y_array_balanced_upsampled_S1S2_WordGo_all_subjects.pkl",
        "rb",
    ) as f:
        Y_array = pickle.load(f)

print("X_array.shape: ", X_array.shape)
Y_hist = np.histogram(Y_array)
Y_hist_sum = Y_hist[0][0] + Y_hist[0][-1]
print(
    "Fluent Trials: {} ({:.2f}%), Stutter Trials: {} ({:.2f}%)".format(
        Y_hist[0][0],
        100 * (Y_hist[0][0] / Y_hist_sum),
        Y_hist[0][-1],
        100 * (Y_hist[0][-1] / Y_hist_sum),
    )
)


def make_balanced(X, Y):
    "Binary label balancer function"
    try:
        # more fluent than stutter trials. choose randomly as much stutter trials from the fluent and concat.
        # np.where(Y_array==0) will give the indices where the array is 0 (fluent).
        # then using np.random.choice, we choose X_array[Y_array==1].shape[0] (stutter) number of samples from the fluent trials.
        # later we concatenate both data points to create a balanced dataset.
        random_data_points = np.random.choice(
            np.where(Y == 0)[0], size=X[Y == 1].shape[0], replace=False
        )
        assert random_data_points.shape[0] == X[Y == 1].shape[0]
        random_data_points = np.concatenate((random_data_points, np.where(Y == 1)[0]))
        random.shuffle(list(random_data_points))

    except ValueError:
        # more stutter than fluent trials. choose randomly as much fluent trials from the stutter and concat.
        random_data_points = np.random.choice(
            np.where(Y == 1)[0], size=X[Y == 0].shape[0], replace=False
        )
        assert random_data_points.shape[0] == X[Y == 0].shape[0]
        random_data_points = np.concatenate((random_data_points, np.where(Y == 0)[0]))
        random.shuffle(list(random_data_points))
    return (X[random_data_points], Y[random_data_points])


def CNN_A(dropout=0.5):
    K.set_image_data_format("channels_first")

    clear_session()

    channels = 17
    timesteps = 87  # upsampled 58 fps

    inputs = Input(shape=(channels, timesteps))

    input_permute = Permute((1, 2), input_shape=(channels, timesteps))(inputs)
    input_reshape = Reshape((1, channels, timesteps))(input_permute)

    conv2d_1 = Conv2D(
        32,
        (1, channels),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(input_reshape)
    conv2d_1_bn = BatchNormalization()(conv2d_1)

    conv2d_2DW = DepthwiseConv2D(
        (channels, 1),
        use_bias=False,
        activation="linear",
        depth_multiplier=2,
        padding="valid",
        kernel_constraint=max_norm(1.0),
    )(conv2d_1_bn)
    conv2d_2DW_bn = BatchNormalization()(conv2d_2DW)
    conv2d_2DW_bn_act = Activation("elu")(conv2d_2DW_bn)

    conv2d_2DW_bn_act_avpool = AveragePooling2D((1, 4))(conv2d_2DW_bn_act)
    conv2d_2DW_bn_act_avpool_dp = Dropout(rate=dropout)(conv2d_2DW_bn_act_avpool)

    conv2d_3Sep = SeparableConv2D(
        32, (1, channels - 1), activation="linear", padding="same"
    )(conv2d_2DW_bn_act_avpool_dp)
    conv2d_3Sep_bn = BatchNormalization()(conv2d_3Sep)
    conv2d_3Sep_bn_act = Activation("elu")(conv2d_3Sep_bn)

    conv2d_3Sep_bn_act_avgpool = AveragePooling2D((1, 8))(conv2d_3Sep_bn_act)
    conv2d_3Sep_bn_act_avgpool_dp = Dropout(rate=dropout)(conv2d_3Sep_bn_act_avgpool)

    flatten_1 = Flatten()(conv2d_3Sep_bn_act_avgpool_dp)
    dense_1 = Dense(
        64, activation="elu", kernel_constraint=max_norm(0.25), name="embedding"
    )(flatten_1)

    predictions = Dense(
        1, activation="sigmoid", kernel_constraint=max_norm(0.25), name="predictions"
    )(dense_1)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def CNN_A2(dropout=0.5):
    K.set_image_data_format("channels_first")

    clear_session()

    channels = 17
    timesteps = 87  # upsampled 58 fps

    inputs = Input(shape=(channels, timesteps))

    input_permute = Permute((1, 2), input_shape=(channels, timesteps))(inputs)
    x = Reshape((1, channels, timesteps))(input_permute)

    x = Conv2D(
        32,
        (1, channels),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        32,
        (1, channels),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)

    x = Dropout(rate=dropout)(x)

    x = Conv2D(
        32,
        (1, channels),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D(
        (channels, 1),
        use_bias=False,
        activation="linear",
        depth_multiplier=2,
        padding="valid",
        kernel_constraint=max_norm(1.0),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = AveragePooling2D((1, 4))(x)
    x = Dropout(rate=dropout)(x)

    x = SeparableConv2D(32, (1, channels - 1), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = AveragePooling2D((1, 8))(x)
    x = Dropout(rate=dropout)(x)

    x = Flatten()(x)
    x = Dense(64, activation="relu", name="embedding")(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def CNN_A3(dropout=0.25):
    K.set_image_data_format("channels_first")

    clear_session()

    channels = 17
    timesteps = 87  # upsampled 58 fps

    inputs = Input(shape=(channels, timesteps))

    input_permute = Permute((1, 2), input_shape=(channels, timesteps))(inputs)
    x = Reshape((1, channels, timesteps))(input_permute)

    x = Conv2D(
        32,
        (1, channels),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        32,
        (1, channels),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        32,
        (channels, 1),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D(
        (channels, 1),
        use_bias=False,
        activation="linear",
        depth_multiplier=2,
        padding="valid",
        kernel_constraint=max_norm(1.0),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = AveragePooling2D((1, 2))(x)
    x = Dropout(rate=dropout)(x)

    x = SeparableConv2D(32, (1, channels - 1), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = AveragePooling2D((1, 2))(x)
    x = Dropout(rate=dropout)(x)

    x = Flatten()(x)

    x = Dense(256, activation="relu", name="d1")(x)

    x = Dense(128, activation="relu", name="d2")(x)

    x = Dense(64, activation="relu", name="embedding")(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    return Model(inputs=inputs, outputs=predictions)


def CNN_B(dropout=0.5):
    K.set_image_data_format("channels_first")

    clear_session()

    channels = 17
    timesteps = 87  # upsampled 58 fps

    inputs = Input(shape=(channels, timesteps))

    input_permute = Permute((1, 2), input_shape=(channels, timesteps))(inputs)
    x = Reshape((1, channels, timesteps))(input_permute)

    x = Conv2D(
        16,
        (3, 3),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Conv2D(64, (3, 3), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Flatten()(x)

    x = Dense(256, activation="relu", name="d1")(x)

    x = Dense(128, activation="relu", name="d2")(x)

    x = Dense(64, activation="relu", name="embedding")(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    return Model(inputs=inputs, outputs=predictions)


def CNN_B2(dropout=0.5):
    K.set_image_data_format("channels_first")

    clear_session()

    channels = 17
    timesteps = 87  # upsampled 58 fps

    inputs = Input(shape=(channels, timesteps))

    input_permute = Permute((1, 2), input_shape=(channels, timesteps))(inputs)
    x = Reshape((1, channels, timesteps))(input_permute)

    x = Conv2D(
        16, (4, 4), activation="elu", input_shape=(channels, timesteps), padding="same"
    )(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(32, (4, 4), activation="elu", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Conv2D(64, (4, 4), activation="elu", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(128, (4, 4), activation="elu", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Flatten()(x)

    x = Dense(256, activation="relu", name="d1")(x)

    x = Dense(128, activation="relu", name="d2")(x)

    x = Dense(64, activation="relu", name="embedding")(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    return Model(inputs=inputs, outputs=predictions)


def CNN_B2_2x2(dropout=0.5):
    K.set_image_data_format("channels_first")

    clear_session()

    channels = 17
    timesteps = 87  # upsampled 58 fps

    inputs = Input(shape=(channels, timesteps))

    input_permute = Permute((1, 2), input_shape=(channels, timesteps))(inputs)
    x = Reshape((1, channels, timesteps))(input_permute)

    x = Conv2D(
        16,
        (2, 2),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(32, (2, 2), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Conv2D(64, (2, 2), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(128, (2, 2), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Flatten()(x)

    x = Dense(256, activation="relu", name="d1")(x)

    x = Dense(128, activation="relu", name="d2")(x)

    x = Dense(64, activation="relu", name="embedding")(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    return Model(inputs=inputs, outputs=predictions)


def CNN_B2_6x6(dropout=0.5):
    K.set_image_data_format("channels_first")

    clear_session()

    channels = 17
    timesteps = 87  # upsampled 58 fps

    inputs = Input(shape=(channels, timesteps))

    input_permute = Permute((1, 2), input_shape=(channels, timesteps))(inputs)
    x = Reshape((1, channels, timesteps))(input_permute)

    x = Conv2D(
        16,
        (6, 6),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(32, (2, 6), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Conv2D(64, (2, 6), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(128, (2, 6), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Flatten()(x)

    x = Dense(256, activation="relu", name="d1")(x)

    x = Dense(128, activation="relu", name="d2")(x)

    x = Dense(64, activation="relu", name="embedding")(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    return Model(inputs=inputs, outputs=predictions)


def CNN_B3(dropout=0.5):
    K.set_image_data_format("channels_first")

    clear_session()

    channels = 17
    timesteps = 87  # upsampled 58 fps

    inputs = Input(shape=(channels, timesteps))

    input_permute = Permute((1, 2), input_shape=(channels, timesteps))(inputs)
    x = Reshape((1, channels, timesteps))(input_permute)

    x = Conv2D(
        16,
        (4, 4),
        activation="linear",
        input_shape=(channels, timesteps),
        padding="same",
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, (4, 4), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Conv2D(32, (4, 4), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (4, 4), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Dropout(rate=dropout)(x)

    x = Conv2D(64, (4, 4), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (4, 4), activation="linear", padding="same")(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(256, activation="relu", name="d1")(x)

    x = Dense(128, activation="relu", name="d2")(x)

    x = Dense(64, activation="relu", name="embedding")(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    return Model(inputs=inputs, outputs=predictions)


# model = CNN_B()
# print(model.summary())


def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()


def plot_auc(fpr, tpr, auc_value):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label="AUC-ROC (area = {:.3f})".format(auc_value))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="best")
    plt.show()


seed = 13
np.random.seed(seed)

X_array_, X_test, Y_array_, y_test = train_test_split(
    X_array, Y_array, test_size=0.1, random_state=seed, shuffle=True
)

kfold = KFold(n_splits=5, random_state=seed)

accuracies = []
precisions = []
recalls = []
f1s = []
aucs = []
count = 1

for train, test in kfold.split(X_array_, Y_array_):
    print("Processing Fold ", count)

    print("Train Data Statistics")
    print("X_array.shape: ", X_array_[train].shape)
    Y_hist = np.histogram(Y_array_[train])
    Y_hist_sum = Y_hist[0][0] + Y_hist[0][-1]
    print(
        "Fluent Trials: {} ({:.2f}%), Stutter Trials: {} ({:.2f}%)".format(
            Y_hist[0][0],
            100 * (Y_hist[0][0] / Y_hist_sum),
            Y_hist[0][-1],
            100 * (Y_hist[0][-1] / Y_hist_sum),
        )
    )

    print("Validation Data Statistics")
    print("X_array.shape: ", X_array_[test].shape)
    Y_hist = np.histogram(Y_array_[test])
    Y_hist_sum = Y_hist[0][0] + Y_hist[0][-1]
    print(
        "Fluent Trials: {} ({:.2f}%), Stutter Trials: {} ({:.2f}%)".format(
            Y_hist[0][0],
            100 * (Y_hist[0][0] / Y_hist_sum),
            Y_hist[0][-1],
            100 * (Y_hist[0][-1] / Y_hist_sum),
        )
    )

    print("Test Data Statistics")
    print("X_array.shape: ", X_test.shape)
    Y_hist = np.histogram(y_test)
    Y_hist_sum = Y_hist[0][0] + Y_hist[0][-1]
    print(
        "Fluent Trials: {} ({:.2f}%), Stutter Trials: {} ({:.2f}%)".format(
            Y_hist[0][0],
            100 * (Y_hist[0][0] / Y_hist_sum),
            Y_hist[0][-1],
            100 * (Y_hist[0][-1] / Y_hist_sum),
        )
    )

    count += 1
    try:
        del model
    except NameError:
        pass

    hyper_parameters = {}
    hyper_parameters["lr"] = 0.01
    hyper_parameters["momentum"] = 0.9
    hyper_parameters["lr_factor"] = 0.5
    hyper_parameters["lr_patience"] = 15
    hyper_parameters["lr_cooldown"] = 10
    hyper_parameters["min_lr"] = 0.000001
    hyper_parameters["early_min_delta"] = 0.01
    hyper_parameters["early_patience"] = 30
    hyper_parameters["batch_size"] = 64
    hyper_parameters["epochs"] = 1000
    hyper_parameters["dropout"] = 0.5
    hyper_parameters["optimizer"] = "sgd"
    hyper_parameters["loss"] = "binary_crossentropy"

    # CW
    hyper_parameters = {}
    hyper_parameters["lr"] = 0.01
    hyper_parameters["momentum"] = 0.9
    hyper_parameters["lr_factor"] = 0.5
    hyper_parameters["lr_patience"] = 15
    hyper_parameters["lr_cooldown"] = 10
    hyper_parameters["min_lr"] = 1e-6
    hyper_parameters["early_min_delta"] = 0.01
    hyper_parameters["early_patience"] = 30
    hyper_parameters["batch_size"] = 64
    hyper_parameters["epochs"] = 1000
    hyper_parameters["dropout"] = 0.5
    hyper_parameters["optimizer"] = "sgd"
    hyper_parameters["loss"] = "binary_crossentropy"

    sgd = optimizers.SGD(
        lr=hyper_parameters["lr"], momentum=hyper_parameters["momentum"], nesterov=True
    )
    adam = optimizers.Adam(learning_rate=hyper_parameters["lr"], amsgrad=False)
    hyper_parameters["optimizer"] = "adam"

    reduce_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=hyper_parameters["lr_factor"],
        patience=hyper_parameters["lr_patience"],
        cooldown=hyper_parameters["lr_cooldown"],
        min_lr=hyper_parameters["min_lr"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=hyper_parameters["early_min_delta"],
        patience=hyper_parameters["early_patience"],
    )

    model = CNN_B2(dropout=hyper_parameters["dropout"])
    model_name = "_CNN_B2_6x6_" + paradigm + "_"

    model.compile(
        loss=hyper_parameters["loss"],
        optimizer=hyper_parameters["optimizer"],
        metrics=["accuracy"],
    )

    history = model.fit(
        X_array_[train],
        Y_array_[train],
        epochs=hyper_parameters["epochs"],
        batch_size=hyper_parameters["batch_size"],
        verbose=1,
        validation_data=(X_array_[test], Y_array_[test]),
        shuffle=True,
        callbacks=[reduce_lr, early_stop],
    )

    plot_history(history)

    test_output = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_binary = np.round(np.clip(y_pred, 0, 1)).flatten()
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    print(
        "Accuracy: {:.2f}%, Recall: {:.3f}, Precision: {:.3f}, F1: {:.3f}".format(
            test_output[1] * 100, recall, precision, f1
        )
    )

    # AUC ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_value = auc(fpr, tpr)
    auc_dict = {}
    auc_dict["fpr"] = fpr
    auc_dict["tpr"] = tpr
    auc_dict["thresholds"] = thresholds
    auc_dict["auc"] = auc_value

    plot_auc(fpr, tpr, auc_value)

    accuracies.append(test_output[1] * 100)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    aucs.append(auc_value)

    if test_output[1] > 0.77 and recall > 0.62:
        model_path = (
            "../trained_models/"
            + str(datetime.date.today())
            + model_name
            + "{:.3f}".format(test_output[1])[-3:]
        )
        print("Saving to: ", model_path + ".h5")
        model.save(model_path + ".h5")
        with open(model_path + "_auc_details.pkl", "wb") as f:
            pickle.dump(auc_dict, f)
        with open(model_path + "_history.pkl", "wb") as f:
            pickle.dump(model.history.history, f)
        with open(model_path + "_params.pkl", "wb") as f:
            pickle.dump(model.history.params, f)
        with open(model_path + "_hyperparameters.pkl", "wb") as f:
            pickle.dump(hyper_parameters, f)

    ##     with open(model_path + '_history.pkl','rb') as f: history2 = pickle.load(f)
    ##     with open(model_path + '_params.pkl','rb') as f: params2 = pickle.load(f)

print("Training Statistics")
metrics_df = pd.DataFrame.from_dict(
    {
        "Accuracy": accuracies,
        "F1": f1s,
        "Recall": recalls,
        "Precision": precisions,
        "AUC": aucs,
    }
)
print(metrics_df.describe())

print("Summary")
for i in [
    "{:.2f}+-{:.2f}".format(metrics_df[metric].mean(), metrics_df[metric].std())
    for metric in metrics_df.columns
]:
    print(i)
