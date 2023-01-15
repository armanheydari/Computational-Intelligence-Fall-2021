# Q2.2_graded
MLP_model = keras.models.Sequential(layers=[
                                            keras.layers.Input(2),
                                            keras.layers.Dense(15, activation='elu'),
                                            keras.layers.Dense(4, activation='softmax')                             
])

MLP_model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

MLP_history = MLP_model.fit(
    x_train,
    y_train,
    epochs = 20000,
    verbose = 0
)

print("MLP accuracy on train set:", MLP_model.evaluate(x_train, y_train)[1] * 100, "%")

# Q2.2_graded
plt.plot(MLP_history.history['accuracy'])
plt.title('MLP accuracy')
plt.show()
plt.plot(MLP_history.history['loss'])
plt.title('MLP loss')
plt.show()

# Q2.2_graded
y_mlp = np.argmax(MLP_model.predict(x_test), axis=1)

plt.scatter(x_test[:, 0], x_test[:, 1], c=y_mlp, cmap='plasma')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('test results with MLP')
plt.show()

