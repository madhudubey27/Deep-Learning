#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
dataset_dir = "C:\Plant_leave_diseases_dataset_without_augmentation"
  
batch_size = 32                 
img_size = (224, 224)            

dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',               
    label_mode='categorical',       
    color_mode='rgb',                
    batch_size=batch_size,
    image_size=img_size,             
    shuffle=True                    
)

for images, labels in dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)


# In[2]:


class_names = dataset.class_names
print("Class names:", class_names)


# In[3]:


dataset_size = tf.data.experimental.cardinality(dataset).numpy()


train_size = int(0.8 * dataset_size) 
val_size = int(0.1 * dataset_size)    
test_size = dataset_size - train_size - val_size  


train_dataset = dataset.take(train_size)
remaining_dataset = dataset.skip(train_size)

val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)


train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


print(f"Total dataset size: {dataset_size}")
print(f"Train dataset size: {train_size}")
print(f"Validation dataset size: {val_size}")
print(f"Test dataset size: {test_size}")


# # vgg16

# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_model_16 = keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3) 
)

base_model_16.trainable = False


inputs = keras.Input(shape=(160, 160, 3))  
x = base_model_16(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)  
x = layers.BatchNormalization()(x)  
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)  
x = layers.Dropout(0.4)(x)  
outputs = layers.Dense(17, activation="softmax")(x)  

model_16 = keras.Model(inputs, outputs)



# In[4]:


model_16.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# In[5]:


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# In[6]:


history = model_16.fit(
    train_dataset,
    epochs=80,  
    validation_data=val_dataset,
    callbacks=[early_stopping]
)


# In[16]:


train_loss, train_accuracy = model_16.evaluate(train_dataset, verbose=1)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")


# In[17]:


val_loss, val_accuracy = model_16.evaluate(val_dataset, verbose=1)
print(f"validation Accuracy: {val_accuracy * 100:.2f}%")


# In[10]:


test_loss, test_accuracy = model_16.evaluate(test_dataset, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# In[7]:


model_16.summary()


# In[12]:


# Accuracy Plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# In[13]:


# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[8]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Predict the labels
y_true_16 = []
y_pred_16 = []

for images, labels in test_dataset:
    y_true_16.extend(np.argmax(labels.numpy(), axis=1)) 
    predictions_16 = model_16.predict(images)
    y_pred_16.extend(np.argmax(predictions_16, axis=1)) 

cm_16 = confusion_matrix(y_true_16, y_pred_16)


# In[11]:


import matplotlib.pyplot as plt
cm_16 = confusion_matrix(y_true_16, y_pred_16)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_16, annot=True, fmt="d", cmap="Blues", xticklabels=range(15), yticklabels=range(15))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# # class balance

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def get_class_distribution(dataset):
    labels = []
    for _, label in dataset:  
        label = label.numpy()  
        if label.ndim > 0:  
            label = label.argmax()  
        labels.append(label)

    class_counts = Counter(labels)  
    return class_counts


class_counts = get_class_distribution(train_dataset)


print("Class Distribution:", class_counts)


# In[5]:


def plot_class_distribution(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(classes, counts, color='skyblue', alpha=0.7)
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution in Dataset")
    plt.xticks(classes)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for i, count in enumerate(counts):
        plt.text(classes[i], count + 5, str(count), ha="center", fontsize=12)

    plt.show()

plot_class_distribution(class_counts)


# class weights

# In[22]:


import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

all_labels = []

for _, label in train_dataset:
    label = label.numpy()

   
    if isinstance(label, np.ndarray) and label.ndim > 1:
        label = np.argmax(label)  
    
    all_labels.append(int(label)) 

all_labels = np.array(all_labels).flatten()

class_labels = np.unique(all_labels)  
print("Detected Classes:", class_labels)  

class_weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=all_labels)

class_weight_dict = {class_labels[i]: class_weights[i] for i in range(len(class_labels))}
print("Computed Class Weights:", class_weight_dict)


# # data augmentation

# In[14]:


from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),  
    RandomRotation(0.2),  
    RandomZoom(0.2),  
])


# In[15]:


train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))


# # resnet50

# In[27]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


# In[28]:


base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output) 
x = Dense(256, activation="relu")(x)           
x = Dropout(0.4)(x)                             
x = Dense(128, activation="relu")(x)            
x = Dropout(0.3)(x)                             
output_layer = Dense(17, activation="softmax")(x)  


model = Model(inputs=base_model.input, outputs=output_layer)

model.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# In[29]:


early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.5, min_lr=1e-6)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=30, 
                    class_weight=class_weights, callbacks=[early_stopping, lr_scheduler])


# In[30]:


# Unfreeze Top 50% of ResNet50 Layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

# againCompilewith a Lower Learning Rate
model.compile(optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Fine-Tune the Model
history_fine = model.fit(train_dataset, validation_data=val_dataset, epochs=30, 
                         class_weight=class_weights, callbacks=[early_stopping, lr_scheduler])


# In[31]:


test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")


# In[33]:


train_loss, train_acc = model.evaluate(train_dataset)
print(f"Train Accuracy: {train_acc:.4f}")


# In[34]:


val_loss, val_acc = model.evaluate(val_dataset)
print(f"Validation Accuracy: {val_acc:.4f}")


# In[32]:


def plot_metrics(history):
    plt.figure(figsize=(12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.show()

plot_metrics(history_fine)


# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

true_labels = []
pred_labels = []

for images, labels in test_dataset:  
    preds = model.predict(images)  
    pred_classes = np.argmax(preds, axis=1)  
    true_classes = np.argmax(labels.numpy(), axis=1) 
    
    true_labels.extend(true_classes)
    pred_labels.extend(pred_classes)

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

cm = confusion_matrix(true_labels, pred_labels)


# In[36]:


plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(17), yticklabels=range(17))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for ResNet50")
plt.show()


# In[37]:


#classification report
class_names = [f"Class {i}" for i in range(17)] 
report = classification_report(true_labels, pred_labels, target_names=class_names)
print("Classification Report:\n", report)


# In[38]:


model.save("resnet50model.h5")


# In[ ]:




