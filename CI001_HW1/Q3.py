# Q3_graded
import tensorflow as tf
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Q3_graded
training_images = training_images/255.0
test_images = test_images/255.0 

# Q3_graded
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape = (28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')
])

# Q3_graded
model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['accuracy']
)
model.summary()

# Q3_graded
history = model.fit(
    training_images, 
    training_labels,
    epochs = 50,
    shuffle=True,
    batch_size=32,
)

# Q3_graded
loss, accuracy = model.evaluate(training_images, training_labels)
print("train accuracy:", accuracy*100, "%")
loss, accuracy = model.evaluate(test_images, test_labels)
print("test accuracy:", accuracy*100, "%")

