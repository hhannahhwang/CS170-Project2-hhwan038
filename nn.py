import numpy as np
import time
from numba import njit

#loads dataset from text file and separates class labels and features
def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1:]

def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1 
    return (features - mean) / std

@njit #nearest neighbor classifier with leave-one-out cross-validation
def nn_classifier(data, feature_indices_to_use):
    number_correctly_classified = 0
    num_instances = data.shape[0]
    
    #empty set case
    if len(feature_indices_to_use) == 0:
        return 0.0 

    #Iterates through every instance
    for i in range(num_instances):
        label_to_classify = data[i, 0]
        nn_distance = np.inf
        nn_label = -1.0

        for k in range(num_instances):
            if i == k: continue
            
            #calculate Euclidean distance across selected features subset
            current_dist_sq = 0.0
            for feat_idx in feature_indices_to_use:
                diff = data[i, feat_idx + 1] - data[k, feat_idx + 1]
                current_dist_sq += diff * diff
                if current_dist_sq >= nn_distance: 
                    break
            
            #update nearest neighbor
            if current_dist_sq < nn_distance:
                nn_distance = current_dist_sq
                nn_label = data[k, 0]

        if label_to_classify == nn_label:
            number_correctly_classified += 1

    return number_correctly_classified / num_instances

def forward_selection(data):
    num_features = data.shape[1] - 1
    current_features = []
    best_overall_acc = 0
    best_overall_features = []

    for _ in range(num_features):
        feature_to_add_at_this_level = -1
        best_acc_at_this_level = 0

        for k in range(num_features):
            if k not in current_features:
                test_set = np.array(current_features + [k], dtype=np.int32)
                accuracy = nn_classifier(data, test_set)
                print(f"   Using feature(s) {{{', '.join(str(f+1) for f in current_features + [k])}}} accuracy is {accuracy * 100:.1f}%")

                if accuracy > best_acc_at_this_level:
                    best_acc_at_this_level = accuracy
                    feature_to_add_at_this_level = k

        if feature_to_add_at_this_level != -1:
            current_features.append(feature_to_add_at_this_level)
            print(f"\nFeature set {{{', '.join(str(f+1) for f in current_features)}}} was best, accuracy is {best_acc_at_this_level * 100:.1f}%")

            if best_acc_at_this_level > best_overall_acc:
                best_overall_acc = best_acc_at_this_level
                best_overall_features = list(current_features)

    print(f"Finished search!! The best feature subset is {{{', '.join(str(f+1) for f in best_overall_features)}}}, which has an accuracy of {best_overall_acc * 100:.1f}%")

def backward_elimination(data):
    num_features = data.shape[1] - 1
    current_features = list(range(num_features))
    
    initial_acc = nn_classifier(data, np.array(current_features, dtype=np.int32))
    best_overall_acc = initial_acc
    best_overall_features = list(current_features)
    print(f"Using all features {{{', '.join(str(f+1) for f in current_features)}}} accuracy is {initial_acc * 100:.1f}%\n")

    for _ in range(num_features - 1):
        feature_to_remove_at_this_level = -1
        best_acc_at_this_level = 0

        for k in current_features:
            temp_set = [f for f in current_features if f != k]
            accuracy = nn_classifier(data, np.array(temp_set, dtype=np.int32))
            print(f"   Using feature(s) {{{', '.join(str(f+1) for f in temp_set)}}} accuracy is {accuracy * 100:.1f}%")

            if accuracy > best_acc_at_this_level:
                best_acc_at_this_level = accuracy
                feature_to_remove_at_this_level = k

        current_features.remove(feature_to_remove_at_this_level)
        print(f"\nFeature set {{{', '.join(str(f+1) for f in current_features)}}} was best, accuracy is {best_acc_at_this_level * 100:.1f}%")

        if best_acc_at_this_level >= best_overall_acc:
            best_overall_acc = best_acc_at_this_level
            best_overall_features = list(current_features)

    print(f"Finished search!! The best feature subset is {{{', '.join(str(f+1) for f in best_overall_features)}}}, which has an accuracy of {best_overall_acc * 100:.1f}%")

# --- Main Execution ---
print("Welcome to Hannah Hwang's Feature Selection Algorithm.")
filename = input("Type in the name of the file to test: ")

labels, features = load_data(filename)
features_norm = normalize_features(features)
data_norm = np.column_stack((labels, features_norm))

all_acc = nn_classifier(data_norm, np.arange(features.shape[1], dtype=np.int32))
print(f"\nThis dataset has {features.shape[1]} features with {len(labels)} instances.\n")
print(f"Running nearest neighbor with all {features.shape[1]} features, using \"leaving-one-out\" evaluation, I get an accuracy of {all_acc * 100:.1f}%\n")

choice = input("Type the number of the algorithm you want to run.\n1) Forward Selection\n2) Backward Elimination\n")
print(f"Beginning Search...")

start_time = time.time()

if choice == '1':
    forward_selection(data_norm)
elif choice == '2':
    backward_elimination(data_norm)

end_time = time.time()
total_duration = end_time - start_time
print(f"\nTotal Execution time: {total_duration:.2f} seconds")