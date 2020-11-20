from sklearn import preprocessing, model_selection
import pandas as pd
import numpy as np
import time

begin = time.time()

data = pd.read_csv("C:\\Users\\Srijib\\.vscode\\machine-learning\\cancer-predict-scratch\\cancer_data.csv")

def normalize(a, max):
    for i in range(len(a)):
        a[i] = a[i]/max
    return a

# Processing and normalizing the data 
le = preprocessing.LabelEncoder()
diagnosis = le.fit_transform(list(data["diagnosis"]))
radius_mean = list(data["radius_mean"])
normalize(radius_mean, max(radius_mean))
texture_mean = list(data["texture_mean"])
normalize(texture_mean, max(texture_mean))
perimeter_mean = list(data["perimeter_mean"])
normalize(perimeter_mean, max(perimeter_mean))
area_mean = list(data["area_mean"])
normalize(area_mean, max(area_mean))
smoothness_mean = list(data["smoothness_mean"])
compactness_mean = list(data["compactness_mean"])
concavity_mean = list(data["concavity_mean"])
concave_points_mean = list(data["concave_points_mean"])
symmetry_mean = list(data["symmetry_mean"])
fractal_dimension_mean = list(data["fractal_dimension_mean"])
radius_se = list(data["radius_se"])
normalize(radius_se, max(radius_se))
texture_se = list(data["texture_se"])
normalize(texture_se, max(texture_se))
perimeter_se = list(data["perimeter_se"])
normalize(perimeter_se, max(perimeter_se))
area_se = list(data["area_se"])
normalize(area_se, max(area_se))
smoothness_se = list(data["smoothness_se"])
compactness_se = list(data["compactness_se"])
concavity_se = list(data["concavity_se"])
concave_points_se = list(data["concave_points_se"])
symmetry_se = list(data["symmetry_se"])
fractal_dimension_se = list(data["fractal_dimension_se"])
radius_worst = list(data["radius_worst"])
normalize(radius_worst, max(radius_worst))
texture_worst = list(data["texture_worst"])
normalize(texture_worst, max(texture_worst))
perimeter_worst = list(data["perimeter_worst"])
normalize(perimeter_worst, max(perimeter_worst))
area_worst = list(data["area_worst"])
normalize(area_worst, max(area_worst))
smoothness_worst = list(data["smoothness_worst"])
compactness_worst = list(data["compactness_worst"])
concavity_worst = list(data["concavity_worst"])
concave_points_worst = list(data["concave_points_worst"])
symmetry_worst = list(data["symmetry_worst"])
fractal_dimension_worst = list(data["fractal_dimension_worst"])

predict = "diagnosis"

X = list(zip(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, 
            symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, 
            concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
            smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst))

y = list(diagnosis)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

'''
Take inputs.
Assign random weights in the hidden layer and the output layer.
Run the code for training.
Find the error in prediction.
Update the weight values of the hidden layer and output layer by gradient descent algorithm.
Repeat the training phase with updated weights.
Make predictions.
'''

# Define input and target output
input_features = np.array(X_train)
target_output = np.array(y_train).reshape(len(y_train), 1)

INPUT_NODES = 30
HIDDEN_NODES = 128
OUTPUT_NODES = 1

weight_hidden = np.random.rand(INPUT_NODES, HIDDEN_NODES)
weight_output = np.random.rand(HIDDEN_NODES, OUTPUT_NODES)

# Learning rate
lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#Logic to update weights
for epoch in range(200000):

    #Input for hidden layer
    input_hidden = np.dot(input_features, weight_hidden)

    # Output from hidden layer
    output_hidden = sigmoid(input_hidden)

    # Input for output layer
    input_op = np.dot(output_hidden, weight_output)
    input_op /= 100

    #Output from output layer
    output_op = sigmoid(input_op)

    # PHASE 1

    # Calculating mean squared error
    error_out = ((1/2) * (np.power((output_op - target_output), 2)))

    # Derivatives for phase 1
    derror_douto = output_op - target_output
    douto_dino = sigmoid_der(input_op)
    dino_dwo = output_hidden

    derror_dwo = np.dot(dino_dwo.T, derror_douto * douto_dino)

    # PHASE 2

    # Derivatives for phase 2
    derror_dino = derror_douto * douto_dino
    dino_douth = weight_output
    derror_douth = np.dot(derror_dino, dino_douth.T)
    douth_dinh = sigmoid_der(input_hidden)
    dinh_dwh = input_features
    derror_wh = np.dot(dinh_dwh.T, douth_dinh * derror_douth)
    
    # Update weights

    weight_hidden -= lr * derror_wh
    weight_output -= lr * derror_dwo

i = 0
correct = 0
for case in X_test:
    result1 = np.dot(case, weight_hidden)
    result2 = sigmoid(result1)
    result3 = np.dot(result2, weight_output)
    result3 /= 100
    result4 = round(sigmoid(result3)[0])
    if result4 == y_test[i]:
        correct += 1
    i += 1

print("Hidden weights:",weight_hidden)
print("Output weight:", weight_output)
accuracy = correct/len(X_test)
print("Accuracy:", accuracy)

end = time.time()
print(f"Total runtime of the program is {end - begin}s")