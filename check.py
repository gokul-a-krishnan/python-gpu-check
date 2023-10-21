import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.test import is_gpu_available
from tensorflow.config import list_physical_devices

print(is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))
print(list_physical_devices('GPU'))


def load_dataset():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_x, train_y, test_x, test_y


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


def define_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def run_test():
    train_x, train_y, test_x, test_y = load_dataset()
    train_x, test_x = prep_pixels(train_x, test_x)
    model = define_model()
    model.fit(train_x, train_y, epochs=5, batch_size=64, validation_data=(test_x, test_y))


run_test()
