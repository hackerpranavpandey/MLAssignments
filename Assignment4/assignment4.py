import numpy as np
import matplotlib.pyplot as plt
num_samples = 100
# True means and covariance matrices of the Gaussians generating the data from two classesclass_1_mean = np.array([1.0, 1.0])
class_1_mean=np.array([1.0,1.0])
class_2_mean = np.array([-2.0, -2.0])
# Let's use non-spherical classes (non-identity covariance matrix for each Gaussian)
class_1_cov = np.array([[0.8, 0.4], [0.4, 0.8]])
class_2_cov = np.array([[0.8, -0.6], [-0.6, 0.8]])
X_class_1 = np.random.multivariate_normal(class_1_mean, class_1_cov, num_samples)
X_class_2 = np.random.multivariate_normal(class_2_mean, class_2_cov, num_samples)
X_train = np.vstack((X_class_1, X_class_2))
y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))
# spliting the dataset into test and train in ration 80:20
X_train_split = np.vstack((X_class_1[:80], X_class_2[:80]))
y_train_split = np.hstack((np.zeros(num_samples-20), np.ones(num_samples-20)))
X_test=np.vstack((X_class_1[80:], X_class_2[80:]))
y_test_split = np.hstack((np.zeros(20), np.ones(20)))

# Scatter plot
plt.scatter(X_class_1[:, 0], X_class_1[:, 1], label='Class 1', marker='o')
plt.scatter(X_class_2[:, 0], X_class_2[:, 1], label='Class 2', marker='x')
# plot for class 1 and class 2
plt.title('Generated Data from Two Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# to learn k we are using minimum error approach at which k error is minimum
import numpy as np
import matplotlib.pyplot as plt
# below is the algotrithm for KNN Classifier
def KNN_Classifier(X_train_split,X_test,y_train_split,k):
    # this will store distance of k nearest point
    y_test=[]
    for j in X_test:
         dist=[]
         label_index=[] # it will return index value corresponding to k nearest neighbour
         # intitialising the initial distance to infinity
         for i in range(0,k):
           dist.append(float('inf'))
           label_index.append(-1)
         index=0
         for i in X_train_split:
           d=((i[0]-j[0])**2+(i[1]-j[1])**2)**0.5
           # updating k nearest point distances
           for l in range(0,k):
             if(dist[l]>d):
               dist[l]=d
               label_index[l]=index
               break
           index+=1
         y=0
         for index in label_index:
            y+=y_train_split[index]
         y=y/k
         if(y>=0.5):
            y_test.append(1)
         else:
            y_test.append(0)
    return np.array(y_test)
# this is MSE function
def error_loss(y_test_split,y_pred):
  min=0
  for i in range(0,len(y_test_split)):
    min+=(y_test_split[i]-y_pred[i])**2
  min_loss=min/len(y_test_split)
  return min_loss


error=float('inf')
k=1
# below algorithmn will train k for KNN classifier
for i in range(1,160):
  y_pred=KNN_Classifier(X_train_split,X_test,y_train_split,i)
  # print(y_pred)
  e=error_loss(y_test_split,y_pred)
  # print(e)
  if(e<=error):
    error=e
    k=i
  else:
    # print("error is too high")
    continue
print('Below is results on KNN classifier for gaussians distribution')
print(f'The value of k for minimum loss is {k}')
print(f'The value of Mean Square Error for k {k} is {error}')

result = KNN_Classifier(X_train_split, X_test, y_train_split,k)
print("Predicted Labels:", result)
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

Z = KNN_Classifier(X_train_split,np.c_[xx.ravel(), yy.ravel()],y_train_split,k)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5)

plt.scatter(X_test[:, 0], X_test[:, 1], c=result, cmap='viridis', marker='x', s=100, linewidths=2, label='Predicted Labels')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K Nearest Neighbors (KNN) Classifier with Decision Boundary')
plt.legend()
plt.show()
# Negative mean function that is for level 0
def negative_mean(X_train_split):
       mean=[0,0]
       for i in range(0,80):
         mean+=X_train_split[i]
         mean=mean/80
       return mean
# Positive mean function that is for level 1
def positive_mean(X_train_split):
       mean=[0,0]
       for i in range(80,160):
         mean+=X_train_split[i]
         mean=mean/80
       return mean
#Learning with Prototype algorithm
def learning_with_prototype(X_train_split,X_test):

          y_pred=[]
          d1=negative_mean(X_train_split)
          d2=positive_mean(X_train_split)
          for i in range(0,len(X_test)):
                f=((d1[0]-X_test[i][0])**2 + (d1[1]-X_test[i][1])**2)-((d2[0]-X_test[i][0])**2 + (d2[1]-X_test[i][1])**2)
                if(f>=0):
                  y_pred.append(1)
                else:
                  y_pred.append(0)
          return np.array(y_pred)
y_pred=learning_with_prototype(X_train_split,X_test)
error=error_loss(y_test_split,y_pred)
print('Below is results on lwp for gaussians distribution')
print(f"The Mean Square Error for learning with prototye is {error}")
result = learning_with_prototype(X_train_split, X_test)
print("Predicted Labels:", result)
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

Z = learning_with_prototype(X_train_split,np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5)

plt.scatter(X_test[:, 0], X_test[:, 1], c=result, cmap='viridis', marker='x', s=100, linewidths=2, label='Predicted Labels')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Learning with Prototype with Decision Boundary')
plt.legend()
plt.show()

#same task for chi-square distribution
import numpy as np
k1 = 7
k2 = 10
num_samples_2 = 100
points_class_1 = np.random.chisquare(k1, (num_samples, 2))
points_class_2 = np.random.chisquare(k2, (num_samples, 2))
X_train_2 = np.vstack((points_class_1, points_class_2))
y_train_2= np.hstack((np.zeros(num_samples), np.ones(num_samples)))
X_train_split_2 = np.vstack((X_class_1[:80], X_class_2[:80]))
y_train_split_2 = np.hstack((np.zeros(num_samples-20), np.ones(num_samples-20)))
X_test_2=np.vstack((X_class_1[80:], X_class_2[80:]))
y_test_split_2 = np.hstack((np.zeros(20), np.ones(20)))
error=float('inf')
# considering initial value of k to be 1
k=1
for i in range(1,160):
  y_pred_2=KNN_Classifier(X_train_split_2,X_test_2,y_train_split_2,i)
  # print(y_pred_2)
  e=error_loss(y_test_split_2,y_pred_2)
  # print(e)
  if(e<=error_loss(y_test_split_2,y_pred_2)):
    error=e
    k=i
  else:
    # print("error is too high")
    continue
print("Below is result on KNN classification for chi-square")
print(f'The value of k for minimum loss is {k}')
print(f"The value of error is {error}")

result = KNN_Classifier(X_train_split_2, X_test_2, y_train_split_2,k)
print("Predicted Labels:", result)
x_min_2, x_max_2 = X_train_2[:, 0].min() - 1, X_train_2[:, 0].max() + 1
y_min_2, y_max_2 = X_train_2[:, 1].min() - 1, X_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min_2, x_max_2, 0.2), np.arange(y_min_2, y_max_2, 0.2))

Z = KNN_Classifier(X_train_split_2,np.c_[xx.ravel(), yy.ravel()],y_train_split_2,k)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5)

plt.scatter(X_test_2[:, 0], X_test_2[:, 1], c=result, cmap='viridis', marker='x', s=100, linewidths=2, label='Predicted Labels')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K Nearest Neighbors (KNN) Classifier with Decision Boundary for chi-square')
plt.legend()
plt.show()

y_pred_2=learning_with_prototype(X_train_split_2,X_test_2)
error=error_loss(y_test_split_2,y_pred_2)
print("Below is result on learning with prototype for chi-square")
print(f"TheMean Square Error for learning with prototype for chi square distribution {error}")
result = learning_with_prototype(X_train_split_2, X_test_2)
print("Predicted Labels:", result)
x_min, x_max = X_train_2[:, 0].min() - 1, X_train_2[:, 0].max() + 1
y_min, y_max = X_train_2[:, 1].min() - 1, X_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

Z = learning_with_prototype(X_train_split_2,np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5)

plt.scatter(X_test_2[:, 0], X_test_2[:, 1], c=result, cmap='viridis', marker='x', s=100, linewidths=2, label='Predicted Labels')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Learning with Prototype with Decision Boundary for chi square')
plt.legend()
plt.show()


# now training for kaggle datset
import pandas as pd
t_shirt=pd.read_csv('TShirt_size.csv')
# mapping M to 0 and N to level 1
t_shirt['T Shirt Size']=t_shirt['T Shirt Size'].map({'M':0,'L':1})
X=t_shirt.drop('T Shirt Size',axis=1)
Y=t_shirt['T Shirt Size']
X=X.to_numpy()
Y=Y.to_numpy()
# spliting the dataset into two parts from each class
X_train=np.vstack((X[0:5],X[7:15]))
X_test=np.vstack((X[5:7],X[14:17]))
Y_train=np.hstack((np.zeros(5),np.ones(8)))
Y_test=np.hstack((np.zeros(2),np.ones(3)))
# as the size of dataset for above random distribution and given dataset from kaggle is different do new function defining for this dataset
def learning_with_prototype_tshirt(X_train,X_test):
          y_pred=[]
          d1=[0,0]
          for i in range(0,5):
            d1+=X_train[i]
          d1=d1/5
          d2=[0,0]
          for i in range(5,13):
            d2+=X_train[i]
          d2=d2/8
          for i in range(0,len(X_test)):
                f=((d1[0]-X_test[i][0])**2 + (d1[1]-X_test[i][1])**2)-((d2[0]-X_test[i][0])**2 + (d2[1]-X_test[i][1])**2)
                if(f>0):
                  y_pred.append(1)
                else:
                  y_pred.append(0)
          return np.array(y_pred)
y_pred_1=learning_with_prototype_tshirt(X_train,X_test)
error_1=error_loss(Y_test,y_pred_1)
print('Kaggle dataset on TShirt size')
print(f"The MSE using learning with protyotype {error_1}")

error_2=float('inf')
k=1

for i in range(1,14):
  y_pred_2=KNN_Classifier(X_train,X_test,Y_train,i)
  # print(y_pred)
  e=error_loss(Y_test,y_pred_2)
  # print(e)
  if(e<=error_2):
    error_2=e
    k=i
  else:
    continue

print(f'The value of k for minimum loss is {k}')
print(f'The MSE for KNN is {error}')
if(error_1>error_2):
  print(f"More error using Learning with prototype {error_1}")
if(error_2>error_1):
  print(f"More error using KNN {error_2}")
if(error_1==error_2):
  print(f"Same error for both which is {error_1}")