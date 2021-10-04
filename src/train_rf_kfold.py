import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle, random
from sklearn.metrics import roc_auc_score, auc
from sklearn.model_selection import KFold

# Balanced
paradigm = "wg"
print("Considering {} paradigm".format(paradigm))
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

seed = 13


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
        random.shuffle(random_data_points)

    except ValueError:
        # more stutter than fluent trials. choose randomly as much fluent trials from the stutter and concat.
        random_data_points = np.random.choice(
            np.where(Y == 1)[0], size=X[Y == 0].shape[0], replace=False
        )
        assert random_data_points.shape[0] == X[Y == 0].shape[0]
        random_data_points = np.concatenate((random_data_points, np.where(Y == 0)[0]))
        random.shuffle(random_data_points)
    return (X[random_data_points], Y[random_data_points])


X_array, Y_array = make_balanced(X_array, Y_array)

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

# Extract the embeddings

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    X_array, Y_array, test_size=0.1, random_state=13
)

kfold = KFold(n_splits=5, random_state=seed)

accuracies = []
precisions = []
recalls = []
f1s = []
aucs = []
count = 1

for train, test in kfold.split(train_features, train_labels):
    print("Training Split {}".format(count))
    count += 1
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=1000, random_state=13)
    # Train the model on training data
    rf.fit(X_array[train].reshape(-1, 17 * 87), Y_array[train])

    # Use the forest's predict method on the test data
    predictions = rf.predict(X_array[test].reshape(-1, 17 * 87))
    # Calculate the absolute errors
    errors = sum(abs(predictions - Y_array[test]))
    # print('Testing accuracy:', round(1-errors/len(test_labels), 2))

    accuracy = 1 - errors / len(Y_array[test])
    precision = precision_score(Y_array[test], predictions)
    recall = recall_score(Y_array[test], predictions)
    f1 = f1_score(Y_array[test], predictions)
    auc_value = roc_auc_score(Y_array[test], predictions)

    accuracies.append(accuracy * 100)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    aucs.append(auc_value)

    print(
        "Accuracy: {:.2f}, F1: {:.2f}, Recall: {:.2f}, Precision: {:.2f}, AUCROC: {:.2f} ".format(
            accuracy, f1, recall, precision, auc_value
        )
    )

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
