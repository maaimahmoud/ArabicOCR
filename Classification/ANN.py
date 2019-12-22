import pickle
import numpy as np

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

class ANN():
    def __init__(self, featuresNumber):
        super().__init__()

        self.x_vals = [] # Features from input images
        self.y_vals = [] # Labels of input images
        
        self.x_train_vals = [] # features for training dataset
        self.y_train_vals = [] # training dataset labels
        
        self.x_test_vals = []  # features for testing dataset
        self.y_test_vals = []  # testing dataset labels

        self.featuresNumber = featuresNumber

    def train(self):

        # print(set(self.y_vals))   
        # n_classes = len(set(self.y_vals))
        # print("n_classes = ", n_classes)
        n_classes = 102

        # Initialising the ANN
        self.classifier = Sequential()

        # Adding the input layer and the first hidden layer
        self.classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = self.featuresNumber))

        self.classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

        # Adding the output layer
        # self.classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        self.classifier.add(Dense(units = n_classes, kernel_initializer = 'uniform', activation = 'softmax'))

        # Compiling the ANN
        self.classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        self.x_train_vals, self.x_test_vals, self.y_train_vals, self.y_test_vals = train_test_split(self.x_vals, self.y_vals, test_size=0.30, random_state=42)
        
        self.y_train_vals = keras.utils.to_categorical(self.y_train_vals, num_classes=n_classes)
        # self.y_test_vals = keras.utils.to_categorical(self.y_test_vals)
        
        self.x_vals_train = np.array(self.x_train_vals)
        self.x_vals_test = np.array(self.x_test_vals)
        self.y_vals_train = np.array(self.y_train_vals)
        self.y_vals_test = np.array(self.y_test_vals)
        
        print("x_vals_train = ", self.x_vals_train.shape, " y_vals_train = ", self.y_vals_train.shape)

        # Fitting the ANN to the Training set
        self.classifier.fit(self.x_vals_train, self.y_vals_train, batch_size = 10, epochs = 50)




    def test(self):
        # Predicting the Test set results
        y_pred = self.classifier.predict(self.x_vals_test)
        # y_pred = (y_pred > 0.5)
        y_pred = np.argmax(y_pred, axis=1)
        # Making the Confusion Matrix
        import sklearn
        # self.y_train_vals = keras.utils.to_categorical(self.y_vals_test)
        # cm = confusion_matrix(self.y_vals_test, y_pred)
        # print(cm)
        # y_pred = self.classifier.predict(self.x_vals_test)
        print(sklearn.metrics.classification_report(y_true=self.y_vals_test, y_pred=list(y_pred)))

    def saveModel(self, fileName):
        # save the model to disk
        # joblib.dump(self.classifier, fileName + '.sav')
        self.classifier.save(fileName + '.h5')
        # pickle.dump(self.classifier, open(fileName + '.sav', 'wb+'))
        # saver = tf.train.Saver()
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # saver.save(sess, 'my_test_model')

    def loadModel(self,fileName):
        self.classifier = keras.models.load_model(fileName+'.h5')
        # load the model from disk
        # self.classifier = joblib.load(fileName + '.sav')
        # self.classifier = pickle.load(open(fileName + '.sav', 'rb'))
        # with tf.Session() as sess:    
        #     saver = tf.train.import_meta_graph('my-model-1000.meta')
        #     saver.restore(sess,tf.train.latest_checkpoint('./'))
        #     print(sess.run('w1:0'))

    def getResult(self, x):
        xVals = np.array(x)
        y_pred = self.classifier.predict(xVals)     
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
        
# if __name__ == "__main__":    
    # images = read_images_in_folder('/home/ahmed/Desktop/LettersDataset')

    # data = len(images)
    # for i in tqdm(range(data)):
    #     image = images[i]
    #     x_vals.append(features_extraction(image, False))
        
    

