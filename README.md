# few-shot-learning-for-rare-data
Rare Data Image Classification System Using Few-Shot Learning

Each source data is as follows.

painting data: [links](https://www.kaggle.com/datasets/steubk/wikiart)[1]

plant data : [links](https://github.com/juhyeok99/few-shot-learning-for-rare-data/tree/main/plant%20dataset/plant%20source%20image)

To confirm the relationship between the image classification performance and the amount of available data, we conducted tests with 1,500 and 100 samples. Plant data were collected through web crawling, matching the number of painting data entries, including 1,000, 500, and 100 samples. We then selected an appropriate number of data for effective image classification to evaluate performance in scenarios involving rare data.

We conducted an experiment to reduce the amount of available data and maximize the features of the rare data. We aimed to confirm the difference between the training results of the model with a large dataset of 1,000 samples and the training results of the rare data. For the cases of 500 and 100 samples, we conducted experiments to compare the classification accuracy between the existing image classification model and the proposed system while gradually reducing the amount of rare data. We tested all experimental datasets with training and test data at an 8:2 ratio. In addition to the augmented system, we also tested the ArtGAN-generated images provided by Wikiart. We further experimented with the ArtGAN data to verify whether the augmented model technology of the proposed system could be generalized and to evaluate the generalization performance of fake image discrimination for the generated images.

We experimented with 1,000 epochs, a batch size of 32, and a learning rate of 0.001 for both the CNN and the proposed system.

## Abstract

Advances in deep learning can address a variety of computer vision problems. In particular, deep learning has shown high performance in image processing. However, large datasets are required to train deep learning models. Previous studies have addressed the problem of data scarcity via the few-shot learning technique. However, a drawback of these studies is that large datasets are required when new tasks are performed. Hence, this study uses data augmentation techniques to address this shortcoming. Furthermore, we propose an image classification system with a few-shot learning technique that achieves high accuracy, even on rare datasets. Compared with traditional image classification models, the proposed system improves classification accuracy by approximately 18% using 100 data points.

## dataset

The dataset has been augmented by the augmentation system proposed in this paper.

[Lee, H.; Kim, H. Object edge-based image generation technique for constructing large-scale image datasets. J. IKEEE 2023, 27, 280–287]

## data pre-processing

**All data used in the experiment follow the following pretreatment.**

**The data experiment was conducted with 1,000 500 and 100 samples, and the experimental data was mixed with source image and augmented image 5:5.**


```
normalization_layer = tf.keras.layers.Rescaling(1./255)

#The process of selecting 1,000 pieces of painting data
#Plant data is excluded because there are fewer data compared to painting data
random_indices = random.sample(painting_indices, 1000)
painting_subset = SubsetRandomSampler(random_indices)


from sklearn.model_selection import train_test_split

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

```

**plant image**

![plant images](https://github.com/user-attachments/assets/cb2a76de-fbd3-4266-9682-42d2cfff0d81)

Plant data were carried out by web crawling.


**painting image**

![image](https://github.com/user-attachments/assets/0582059e-a1ab-4bbc-b5ea-f6f1e1e8e089)

painting data was provided by wikiart.

### References
[1]	Saleh, B.; Elgammal, M. Large-scale classification of fine-art paintings: Learning the right metric on the right feature. CoRR 2015, abs/1505.00855.
