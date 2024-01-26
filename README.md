# Emoji Classification : 😄🧡🍴⚾😞


# Description :
Here , we have used  [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) pre-trained word vectors To sum the feature vector corresponding to each word in each sentence . <br/>
We summed the feature vectors of all of the words in every sentence .
As a result, we have a single summed feature vector for 'each sentence' of the test and train files. <br/>
Finally, we feed the summed feature vectors corresponding to each sentence , to the network , as new form of X_train and X_test .

# How to install 
```
pip -r install requirements.txt
```

# How to run :
you should write your desired sentence and dimension like below command :

```
python emoji_classification.py --sentence "I Love AI" --dimension 200  
```


# Results :

## without Dropout : 

| Feature vectors Dimension   | Train Loss  | Train Accuracy   | Test Loss  | Test Accuracy   | Inference time  |
| :-------------: | ------------- | ------------- | ------------- | ------------- | ------------- |
| 50d  | 0.47  | 0.89  | 0.56  | 0.87  | 0.0460 second  |
| 100d | 0.33  | 0.94  | 0.5   | 0.83  | 0.0468 second  |
| 200d | 0.16  | 0.98  | 0.43  | 0.80  | 0.0474 second  |
| 300d | 0.09  | 1.0   | 0.39  | 0.82  | 0.0489 second  |

 With Dropout = 0.4 :

| Feature vectors Dimension   | Train Loss  | Train Accuracy   | Test Loss  | Test Accuracy   |Inference time  |
| :-------------: | ------------- | ------------- | ------------- | ------------- |------------- |
| 50d  | 0.79  | 0.70  | 0.71  | 0.78  | 0.0458 second
| 100d | 0.64  | 0.78  | 0.63  | 0.82  | 0.0476 second
| 200d | 0.36  | 0.89  | 0.45  | 0.83  | 0.049  second
| 300d | 0.20  | 0.95  | 0.44  | 0.85  | 0.0498 second

+ ## Dropout conclusion :

+ ### 🔺 Using Dropout causes higher values in "Test Accuracy" in higher dimensions;and resulting in better performance on unseen data(in larger dimensions)  

+ ### 🔻 But instead with Dropout , "Train accuracy" values have been decreased in all dimensions . 

+ ### ⚠ Using Dropout , increases both train and test loss values .