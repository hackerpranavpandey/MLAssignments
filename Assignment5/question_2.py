import pandas as pd
import numpy as np
drug_file=pd.read_csv('drug200.csv')
# print(drug_file)
## mapping F to 0 M to 1
drug_file['Sex']=drug_file['Sex'].map({'F':0,'M':1})
## mapping low to 0 high to 2 normal to 1
drug_file['BP']=drug_file['BP'].map({'LOW':0,'NORMAL':1,'HIGH':2})
drug_file['Cholesterol']=drug_file['Cholesterol'].map({'LOW':0,'NORMAL':1,'HIGH':2})
drug_file['Drug']=drug_file['Drug'].map({'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4})
## droping the Na_to_K feature as it is said in assignment
## but after droping this featur the accuracy comes down to 0.6 but
drug_file=drug_file.drop('Na_to_K',axis=1)
np.max(drug_file['Age']),np.min(drug_file['Age'])
## mapping age to 0,1 and 2 based on their value ranges like here
## if it is less than 30 then map it to 0
## for between 30 and 50 map it to 1
## when greater than 50 map it to 2
def map_age(age):
  if(age<30):
    return 0;
  elif(age>=30 and age<50):
    return 1;
  else:
    return 2;
drug_file['Age']=drug_file['Age'].apply(map_age)
## now lets scale down Na_to_K to scale of 0 and 1 using Z-Score Normalisation
## just few maths to analyse dataset

X = drug_file.iloc[:, :-1].values
Y = drug_file.iloc[:, -1].values.reshape(-1,1)
## dataset split function that will return x_train,y_train,x_test,y_test
def split_dataset(X, Y, split_ratio=0.8):
    ## fixed seed value help split dataset better
    np.random.seed(42)
    ## some shuffling in values
    shuffled_indices = np.random.permutation(len(X))
    train_size = int(len(X) * split_ratio)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    return X_train, Y_train, X_test, Y_test
## split ration 80:20 is better here as size is not large
X_train, Y_train, X_test, Y_test = split_dataset(X, Y, split_ratio=0.8)
# print(Y_train.shape)

## nodes help in making tree and store its left and right child information
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split, max_depth):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
  ## computing entropy with formula discussed in class
    def entropy(self, x_1):
      labels= np.unique(x_1)
      counts = np.bincount(x_1)
      entropy = 0
      for count in counts:
        e_1 = count / len(x_1)
        if e_1 != 0:
            entropy += (-e_1 * np.log2(e_1))
      return entropy

  ## information gain computation is done here
    def information_gain(self, node, left, right):
        w_l = len(left)
        w_r = len(right)
        w_l /= len(node)
        w_r /= len(node)
        g_1 = self.entropy(node)
        g_2 = self.entropy(left)
        g_3 = self.entropy(right)
        inf_gain = g_1 - (w_l * g_2 + w_r * g_3)
        return inf_gain

    def split(self,dataset, feature_index, threshold):
    # Initialize empty lists for left and right subsets
        dataset_left = []
        dataset_right = []
    # Loop through each row in the dataset
        for row in dataset:
        # Check if the feature value of the current row is less than or equal to the threshold
           if row[feature_index] <= threshold:
            # add the row to the left subset
               dataset_left.append(row)
           else:
            #add the row to the right subset
               dataset_right.append(row)
        dataset_left = np.array(dataset_left)
        dataset_right = np.array(dataset_right)
    # Return the left and right subsets
        return dataset_left, dataset_right

    def get_best_split(self, dataset):
      ## this set will give series of feature for spliting at each point or height in tree
        best_split = {}
        max_info_gain = -float("inf")
        num_samples, num_features = dataset.shape
        for feature_index in range(num_features - 1):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
              ## computing lefy and right subtree best split by giving it to split function
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    ## update max_info _gain value for further comparison
                    if curr_info_gain > max_info_gain:
                        best_split = {"feature_index": feature_index,"threshold": threshold,"dataset_left": dataset_left,"dataset_right": dataset_right,"info_gain":curr_info_gain}
                        max_info_gain = curr_info_gain
        return best_split

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset)
            ## checking the information gain here
            if best_split.get("info_gain", 0) > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return {"feature_index": best_split["feature_index"],
                        "threshold": best_split["threshold"],
                        "left": left_subtree,
                        "right": right_subtree}

        return {"value": self.calculate_leaf_value(Y)}
    ## it will compute leaf node for further comparisons
    def calculate_leaf_value(self, Y):
        return max(Y, key=list(Y).count)
    ## build tree with given train dataset
    def fit(self, X, Y):
        Y = np.array(Y).reshape(-1, 1)
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        return [self.make_prediction(x, self.root) for x in X]

    def make_prediction(self, x, tree):
        if "value" in tree:
            return tree["value"]
        feature_val = x[tree["feature_index"]]
        child = tree["left"] if feature_val <= tree["threshold"] else tree["right"]
        return self.make_prediction(x, child)

## object for class DecisionTreeClassifier
Classifier= DecisionTreeClassifier(min_samples_split=5, max_depth=5)
Classifier.fit(X_train,Y_train)
Y_pred = Classifier.predict(X_test)
## below is simple error function that computer error for predicted and real value
def error(y_test,y_pred):
  y_test=np.array(y_test)
  y_pred=np.array(y_pred)
  err=0
  for i in range(0,len(y_test)):
    if(y_pred[i]!=y_test[i]):
      err+=1
  err/=len(y_test)
  return err
e=error(Y_test,Y_pred)
mapping = {4: 'Y', 3: 'X', 2: 'C', 1: 'B', 0: 'A'}
Y_pred = [mapping[val] for val in Y_pred]
print(f'The value of drugs predicted by decision tree is {Y_pred}')
print(f'The value of error with given dataset is for decision tree classification is {e}')
accuracy_decision_tree=1-e
print(f'The accuracy of decision tree is {accuracy_decision_tree}')
print("The accuray that I am getting is droping feature Na_to_K but if that feature is not dropped then I am getting more accuracy than this")

## now implementing it using logistic regression
## but the issue here is that we need to bring each y to scale of 0 and 1
## one way is to split the dataset into the classes they belong
## here total 5 classes are there so we need to split the dataset into five
## for which it comes out to be maximum we assign that class to that dataset
## but this makes it too time taking
## since softmax regression was not taugh tin class implementing using linear regression
# print("different types of label is",np.unique(Y_train))
def generate_y(k):
    y_train=np.zeros(160)
    for i in range(0,160):
        if(Y_train[i]==k):
           y_train[i]=1
    return np.array(y_train)
## for each label computing y_train so that we can apply logistic regression five times
Y_0=generate_y(0)
Y_1=generate_y(1)
Y_2=generate_y(2)
Y_3=generate_y(3)
Y_4=generate_y(4)

def sigmoid(z):
      return 1/(1+np.exp(-z))
  ## so idea is to apply logistic regression five times
  ## compute weight and bias for each time
  ## apply this to test dataset and compute sigmoid fuction
  ## for which label it give maximum value assign that label to that particular data
  ## simple logistic regression class
class LogisticRegression():
  def __init__(self,alfa=0.001,n_itr=10000):
    self.alfa=alfa
    self.n_itr=n_itr
    self.weights=None
    self.bias=None
    ## below function will compute
  def findWeight(self,X,y):
    n_samples,n_features=X.shape
    ## below is w vector
    self.weight=np.zeros(n_features)
    self.bias=0
    for i in range(0,self.n_itr):
      linear_pred=np.dot(X,self.weight)+self.bias
      pred=sigmoid(linear_pred)
      w_grad=(1/n_samples)*np.dot(X.T,(pred-y))
      b_grad=np.sum(pred-y)
      self.weight=self.weight-self.alfa*w_grad
      self.bias=self.bias-self.alfa*b_grad
  def prediction(self,X):
      linear_pred=np.dot(X,self.weight)+self.bias
      pred=sigmoid(linear_pred)
      ## retrun sigmoid function value instead of final logistic function array of 0 and 1
      return np.array(pred)
## computing weight and bias for drug y=0
lr_0=LogisticRegression()
lr_0.findWeight(X_train,Y_0)
z_0=lr_0.prediction(X_test)
## for lable drug 1
lr_1=LogisticRegression()
lr_1.findWeight(X_train,Y_1)
z_1=lr_1.prediction(X_test)
## similary for label 2
lr_2=LogisticRegression()
lr_2.findWeight(X_train,Y_2)
z_2=lr_2.prediction(X_test)
lr_3=LogisticRegression()
lr_3.findWeight(X_train,Y_3)
z_3=lr_3.prediction(X_test)
lr_4=LogisticRegression()
lr_4.findWeight(X_train,Y_4)
z_4=lr_4.prediction(X_test)
## now this function will compare sigmoid value for each label which return maximun it will assign that label to it
def finalLabel(z_0, z_1, z_2, z_3, z_4):
    y_pred = []
    for i in range(40):
        max_value = max(z_0[i], z_1[i], z_2[i], z_3[i], z_4[i])
        if max_value == z_0[i]:
            y_pred.append(0)
        elif max_value == z_1[i]:
            y_pred.append(1)
        elif max_value == z_2[i]:
            y_pred.append(2)
        elif max_value == z_3[i]:
            y_pred.append(3)
        else:
            y_pred.append(4)
    return np.array(y_pred)
y_prediction = finalLabel(z_0, z_1, z_2, z_3, z_4)
e = error(Y_test, y_prediction)
accuracy_logistic = 1 - e
mapping = {4: 'Y', 3: 'X', 2: 'C', 1: 'B', 0: 'A'}
y_prediction=[mapping[val] for val in y_prediction]
print("Predictions using logistic regression is ", y_prediction)
print(f"The value of accuracy using logistic regression is: {accuracy_logistic}")
if(accuracy_decision_tree>accuracy_logistic):
  print("hence decision tree is more accurate")
else:
  print('hence logistic regression is more accurate')
print('hence we can use both but obviously decision tree is better algorithm')