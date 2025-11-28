import tensorflow as tf
from jax import grad, random
from jax import jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# CARGA DE FASHION-MNIST 

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Tomar únicamente el 10% del conjunto de entrenamiento

porcentaje = 0.10
n_total = train_images.shape[0]
n_reducido = int(n_total * porcentaje)

np.random.seed(0)
indices = np.random.choice(n_total, n_reducido, replace=False)

train_images = train_images[indices]
train_labels = train_labels[indices]

# NORMALIZACIÓN — igual que base

train_images = train_images / 255.0
test_images = test_images / 255.0


train_images = jnp.array(train_images.reshape(-1, 28 * 28))  
test_images  = jnp.array(test_images.reshape(-1, 28 * 28))
train_labels = jnp.array(train_labels)
test_labels  = jnp.array(test_labels)


# DEFINICIÓN DEL MLP 

@jit
def relu(z):
    return jnp.maximum(0, z)

@jit
def softmax(z):
    exp_z = jnp.exp(z)
    return exp_z / jnp.sum(exp_z, axis=-1, keepdims=True)

def mlp(params, x):
    w1, b1, w2, b2, w3, b3 = params
    z1 = jnp.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = jnp.dot(a1, w2) + b2
    a2 = relu(z2)
    z3 = jnp.dot(a2, w3) + b3
    return softmax(z3)

def inicializar_pesos(rng, input_size, hidden_sizes, output_size):
    w1 = random.normal(rng, (input_size, hidden_sizes[0])) * 0.01
    b1 = jnp.zeros((hidden_sizes[0],))
    w2 = random.normal(rng, (hidden_sizes[0], hidden_sizes[1])) * 0.01
    b2 = jnp.zeros((hidden_sizes[1],))
    w3 = random.normal(rng, (hidden_sizes[1], output_size)) * 0.01
    b3 = jnp.zeros((output_size,))
    return w1, b1, w2, b2, w3, b3

input_size = 28 * 28
hidden_sizes = [300, 100]
output_size = 10

rng = random.PRNGKey(0)
params = inicializar_pesos(rng, input_size, hidden_sizes, output_size)


# FUNCIÓN DE PÉRDIDA, ACCURACY Y ACTUALIZACIÓN 

@jit
def cross_entropy_loss(params, x, y):
    preds = mlp(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(preds), axis=-1))

@jit
def accuracy(params, x, y):
    predictions = jnp.argmax(mlp(params, x), axis=1)
    return jnp.mean(predictions == y)

@jit
def actualizar_pesos(params, x, y, learning_rate=0.01):
    grads = grad(cross_entropy_loss)(params, x, y)
    params = [w - learning_rate * dw for w, dw in zip(params, grads)]
    return params


# ENTRENAMIENTO — SIN CAMBIOS

epochs = 30
learning_rate = 0.01
batch_size = 64
accuracies = []

for epoch in range(epochs):
    num_batches = len(train_images) // batch_size

    for i in range(num_batches):
        x_batch = train_images[i*batch_size:(i+1)*batch_size]
        y_batch = jnp.eye(10)[train_labels[i*batch_size:(i+1)*batch_size]]

        params = actualizar_pesos(params, x_batch, y_batch, learning_rate)
    
    test_acc = accuracy(params, test_images, test_labels)
    accuracies.append(test_acc)
    print(f"Época {epoch+1}, Precisión en test: {test_acc:.3f}")


plt.plot(np.arange(len(accuracies)) + 1, accuracies)
plt.xlabel("Época")
plt.ylabel("Precisión en el conjunto de prueba")
plt.title("Accuracy usando solo el 10% del entrenamiento")
plt.show()

# Al tener un menor volumen de datos la curva de presición es distinta y alcanza un máximo menor al pasar las 30 epocas.
