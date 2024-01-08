import tensorflow as tf
import tensorflow_datasets as tfds
from keras_unet.models import custom_unet

dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
train_dataset = dataset["train"]


def preprocess_data(ex):
    image = tf.image.resize(ex["image"], (128, 128))
    mask = tf.image.resize(ex["segmentation_mask"], (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0), tf.expand_dims(mask, axis=0)


train_dataset = train_dataset.map(preprocess_data)

model = custom_unet(
    input_shape=(128, 128, 3),
    num_classes=1,
    filters=64,
    use_batch_norm=True,
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model for a few epochs
model.fit(train_dataset, epochs=1)

validation_dataset = dataset["test"].map(preprocess_data)

model.evaluate(validation_dataset)

for example in validation_dataset.take(1):
    input_image, true_mask = example
    input_image = tf.expand_dims(input_image[0], axis=0)
    predicted_mask = model.predict(input_image)

