#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


class_names = dataset.class_names
print("Class names:", class_names)


# In[4]:


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


# # class balancing

# In[5]:


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


# In[6]:


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


# In[7]:


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

all_labels = []
for _, labels in train_dataset:
    all_labels.extend(np.argmax(labels.numpy(), axis=1)) 

all_labels = np.array(all_labels)

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(all_labels),
    y=all_labels
)

class_weights = {i: class_weights_values[i] for i in np.unique(all_labels)}

print("Computed Class Weights:", class_weights)


# # data augmentation

# In[8]:


from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),  
    RandomRotation(0.2),  
    RandomZoom(0.2),  
])


# In[9]:


train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))


# # vgg19

# In[13]:


from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf


# In[14]:


base_model_v = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

base_model_v.trainable = False

x = GlobalAveragePooling2D()(base_model_v.output)  
x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x) 
x = Dropout(0.5)(x) 
x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)
output = Dense(17, activation="softmax")(x)  

model_v = Model(inputs=base_model_v.input, outputs=output)


# In[17]:


model_v.compile(optimizer=AdamW(learning_rate=1e-3),  
              loss="categorical_crossentropy",  
              metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6)


# In[12]:


history = model_v.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=30, 
                    class_weight=class_weights, 
                    callbacks=[early_stopping, lr_scheduler])


# In[21]:


# Unfreeze Last 80 Layers for Fine-Tuning
for layer in base_model_v.layers[-80:]:  
    layer.trainable = True 

# Compile again with a lower learning rate
model_v.compile(optimizer=AdamW(learning_rate=1e-5),
              loss="categorical_crossentropy", 
              metrics=["accuracy"])


# In[22]:


fine_tune_history = model_v.fit(train_dataset, 
                              validation_data=val_dataset, 
                              epochs=30,  
                              class_weight=class_weights, 
                              callbacks=[early_stopping, lr_scheduler])


# In[15]:


model_v.save("vgg19.h5")


# In[17]:


model_v.save('my_model.keras')


# In[20]:


from tensorflow.keras.models import load_model

# Load the model
model_v = load_model("my_model.keras")

# Check model summary
model_v.summary()


# In[24]:


model_v.save("vgg19.keras")


# In[25]:


test_loss, test_acc = model_v.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")


# In[26]:


train_loss, train_acc = model_v.evaluate(train_dataset)
print(f"Train Accuracy: {train_acc:.4f}")


# In[27]:


val_loss, val_acc = model_v.evaluate(val_dataset)
print(f"Validation Accuracy: {val_acc:.4f}")


# In[28]:


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

plot_metrics(fine_tune_history)


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

true_labels = []
pred_labels = []

for images, labels in test_dataset:  
    preds = model_v.predict(images)  
    pred_classes = np.argmax(preds, axis=1)  
    true_classes = np.argmax(labels.numpy(), axis=1) 
    
    true_labels.extend(true_classes)
    pred_labels.extend(pred_classes)

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

cm = confusion_matrix(true_labels, pred_labels)


# In[30]:


plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(17), yticklabels=range(17))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for ResNet50")
plt.show()


# In[31]:


#classification report
class_names = [f"Class {i}" for i in range(17)] 
report = classification_report(true_labels, pred_labels, target_names=class_names)
print("Classification Report:\n", report)







