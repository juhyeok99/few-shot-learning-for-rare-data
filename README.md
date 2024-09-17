# few-shot-learning-for-rare-data
Rare Data Image Classification System Using Few-Shot Learning

Each source data is as follows.
painting data: links
plant data : https://github.com/juhyeok99/few-shot-learning-for-rare-data/tree/main/plant%20dataset/plant

## Abstract

Advances in deep learning can address a variety of computer vision problems. In particular, deep learning has shown high performance in image processing. However, large datasets are required to train deep learning models. Previous studies have addressed the problem of data scarcity via the few-shot learning technique. However, a drawback of these studies is that large datasets are required when new tasks are performed. Hence, this study uses data augmentation techniques to address this shortcoming. Furthermore, we propose an image classification system with a few-shot learning technique that achieves high accuracy, even on rare datasets. Compared with traditional image classification models, the proposed system improves classification accuracy by approximately 18% using 100 data points.

## dataset

The dataset has been augmented by the augmentation system proposed in this paper.

[Lee, H.; Kim, H. Object edge-based image generation technique for constructing large-scale image datasets. J. IKEEE 2023, 27, 280–287]

## data pre-processing

```
normalization_layer = tf.keras.layers.Rescaling(1./255)

random_indices = random.sample(painting_indices, 1000)
painting_subset = SubsetRandomSampler(random_indices)


from sklearn.model_selection import train_test_split

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

```

**plant image**

![plant images](https://github.com/user-attachments/assets/cb2a76de-fbd3-4266-9682-42d2cfff0d81)

Plant data were carried out by web crawling.


**painting image**

![image](https://github.com/user-attachments/assets/0582059e-a1ab-4bbc-b5ea-f6f1e1e8e089)

painting data was provided by wikiart.
