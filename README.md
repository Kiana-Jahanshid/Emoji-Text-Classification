# Emoji Classification : 😄🧡🍴⚾😞


# Description :


# How to install 
```
pip -r install requirements.txt
```

# How to run :
```
python emoji_classification.py  
```


# Results :

### Without Dropout :

| Feature vectors Dimension   | Train Loss  | Train Accuracy   | Test Loss  | Test Accuracy   | Inference time  |
| :-------------: | ------------- | ------------- | ------------- | ------------- | ------------- |
| 50d  | 0.47  | 0.89  | 0.56  | 0.87  | Content   |
| 100d | 0.33   | 0.94   | 0.5   | 0.83   | Content   |
| 200d  | 0.16   | 0.98   | 0.43   | 0.80   | Content   |
| 300d  | 0.09   | 1.0   | 0.39   | 0.82   | Content   |


## With Dropout = 0.4 :

| Feature vectors Dimension   | Train Loss  | Train Accuracy   | Test Loss  | Test Accuracy   | Inference time  |
| :-------------: | ------------- | ------------- | ------------- | ------------- | ------------- |
| 50d  | 0.79  | 0.70  | 0.71  | 0.78  | Content   |
| 100d | 0.64   | 0.78   | 0.63   | 0.82   | Content   |
| 200d  | 0.36   | 0.89   | 0.45   | 0.83   | Content   |
| 300d  | 0.20   | 0.95   | 0.44   | 0.85   | Content   |


+ ## Dropout conclusion :
```
🔺 Using Dropout causes higher values in "Test Accuracy" in higher dimensions;and resulting in better performance on unseen data(in larger dimensions)  

🔻 But instead , "Train accuracy" values have been decreased in all dimensions versus first table . 
```