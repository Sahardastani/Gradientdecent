import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def Gradient_Descent(Input, Target, Weights, Learning_Rate, Dimen, Iter):
    Input_Transpose = Input.transpose()
    SSE_Array = np.zeros(Iter)
    for in_Iter in range(0, Iter):
        Predicted_Target = np.dot(Input, Weights)
        Error = Predicted_Target - Target
        SSE = np.sum(Error ** 2)
        gradient = np.dot(Input_Transpose, Error) / Dimen
        Weights = Weights - Learning_Rate * gradient
        SSE_Array[in_Iter] = SSE
        print("iteration number %d with SSE: %f" % (in_Iter, SSE))
    return Weights, SSE_Array


# Initialization
Data_Raw = np.load('data.npz')
x1 = Data_Raw.f.x1
x1_test = Data_Raw.f.x1_test
x2 = Data_Raw.f.x2
x2_test = Data_Raw.f.x2_test
y = Data_Raw.f.y
y_test = Data_Raw.f.y_test
Gradien_Order = 1
# make ready x
Bias_Train = np.ones([np.shape(x1)[0], Gradien_Order])
Bias_Test = np.ones([np.shape(x1_test)[0], Gradien_Order])
x_train = np.column_stack((np.multiply(x1, x2 ** 2), x2 ** 2, x1, Bias_Train))
x_test = np.column_stack((np.multiply(x1_test, x2_test ** 2), x2_test ** 2, x1_test, Bias_Test))
# apply gradient
Sample_Size, Dimen = np.shape(x_train)
Iter = 10000
Learning_Rate = 0.0000002
Weights = np.ones(Dimen)
Weights, SSE_Array = Gradient_Descent(x_train, y, Weights, Learning_Rate, Sample_Size, Iter)
# check result
print('Weights array is : ' + str(Weights))
y_p_test = Weights[0] * x_test[:, 0] + Weights[1] * x_test[:, 1] + Weights[2] * x_test[:, 2] + Weights[3] * x_test[:, 3]
Error = y_p_test - y_test
SSE_Test = np.sum(Error ** 2)
print("Final SSE Test is : " + str(SSE_Test))
