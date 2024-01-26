import time 
import pandas as pd 
import tensorflow as tf
import numpy as np 
import argparse

class EmojiTextClassifier :
    def __init__(self , arg):
        self.dimension = arg.dimension

    def load_dataset(self , file_path):
        df = pd.read_csv(file_path)
        sentence = df["sentence"].to_numpy()
        label = df["label"].to_numpy().astype(int)
        return sentence , label

    def load_feature_vectors(self , FeatureVectorsFile):
        FeatureVectors_txt_file = open(FeatureVectorsFile , encoding="utf-8")
        glove_word_vectors={}
        for line in FeatureVectors_txt_file :
            line = line.strip().split()
            word = line[0]
            vector = np.array(line[1:] , dtype=np.float64)
            glove_word_vectors[word] = vector
        return glove_word_vectors
    
    def sentence_to_feature_vectors_avg(self , sentence , glove_word_vectors):
        sentence = sentence.lower()
        words_in_test_sentence =  sentence.strip().split()
        if '"' in words_in_test_sentence :
            words_in_test_sentence.remove('"')
        sum_of_features = np.zeros((self.dimension,))
        for word in words_in_test_sentence :
            sum_of_features +=  glove_word_vectors[word]
        Average_Vector = sum_of_features / len(words_in_test_sentence)
        return Average_Vector

    def load_model(self):
        self.model = tf.keras.models.Sequential([
            #tf.keras.layers.Dropout(0.4, input_shape=(self.dimension,)),
            tf.keras.layers.Dense(5 , input_shape=(self.dimension,) , activation="softmax")
        ])
        return self.model
    

    def train(self , X_train , Y_train , glove_word_vectors):
        X_train_Average_FV=[]
        for x_train in X_train :
            X_train_Average_FV.append(self.sentence_to_feature_vectors_avg(x_train , glove_word_vectors))
        X_train_Average_FV = np.array(X_train_Average_FV)
        Y_train_OneHot = tf.keras.utils.to_categorical(Y_train , num_classes=5)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.categorical_crossentropy , 
                      metrics="accuracy")
        self.model.fit(X_train_Average_FV , Y_train_OneHot , epochs=500)


    def test(self , X_test , Y_test , glove_word_vectors):
        X_test_Average_FV=[]
        for x_test in X_test :
            X_test_Average_FV.append(self.sentence_to_feature_vectors_avg(x_test , glove_word_vectors))
        X_test_Average_FV = np.array(X_test_Average_FV)
        Y_test_OneHot = tf.keras.utils.to_categorical(Y_test , num_classes=5)
        scores = self.model.evaluate(X_test_Average_FV , Y_test_OneHot )
        print("Accuracy :" , scores[1])


    def predict(self , test_sentence , glove_word_vectors):
        Average_FV = self.sentence_to_feature_vectors_avg(test_sentence , glove_word_vectors)
        predict_result = self.model.predict(np.array([Average_FV]))
        y_pred = np.argmax(predict_result)
        emoji = self.label_to_emoji(y_pred)
        print(f"the related emoji to '{test_sentence}' sentence is : " ,  emoji )
    
    
    def label_to_emoji(self , label):
        emojies = ["💚" , "🏀" , "😄" , "😞" , "🍽"]
        emoji = emojies[label]
        return emoji

    
if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence" , type=str , help="write your own sentence")
    parser.add_argument("--dimension" , type=int , help="write dimension of vectors : 50/100/200/300")
    arg= parser.parse_args()
    
    obj = EmojiTextClassifier(arg)
    X_train , Y_train = obj.load_dataset("dataset/train.csv")
    X_test  , Y_test  = obj.load_dataset("dataset/test.csv")

    path = f"glove.6B/glove.6B.{arg.dimension}d.txt"
    glove_word_vectors = obj.load_feature_vectors(path)
    obj.load_model()
    obj.train(X_train , Y_train , glove_word_vectors)
    
    print("\nEVALUATION : ")
    obj.test(X_test ,Y_test , glove_word_vectors)
    
    print("\nPREDICTION : ")
    total_time = 0 
    user_sentence = arg.sentence
    start = time.time()
    for i in range(100):
        obj.predict( user_sentence , glove_word_vectors)
    inference_time = time.time() - start  
    total_inference_time = inference_time / 100
    print("Inference time : " , total_inference_time)


# python emoji_classification.py --sentence "i hate eating fish" --dimension 200  