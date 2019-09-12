from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, save=True, base_name='fig'):
    for i in range(len(history)):
        plt.plot(history[i].history['loss'], label='train loss %02d' % (i+1))
    plt.legend()
    if save:
        plt.savefig(base_name+'_train_loss.png', dpi=300)
    plt.show()

    for i in range(len(history)):
        plt.plot(history[i].history['acc'], label='train acc %02d' % (i+1))
    plt.legend()
    if save:
        plt.savefig(base_name+'_train_acc.png', dpi=300)
    plt.show()

    for i in range(len(history)):
        plt.plot(history[i].history['val_loss'], label='val loss %02d' % (i+1))
    plt.legend()
    if save:
        plt.savefig(base_name+'_val_loss.png', dpi=300)
    plt.show()

    for i in range(len(history)):
        plt.plot(history[i].history['val_acc'], label='val acc %02d' % (i+1))
    plt.legend()
    if save:
        plt.savefig(base_name+'_val_acc.png', dpi=300)
    plt.show()

def plot_history_per_split(history, save=True, base_name='fig'):
    for i in range(len(history)):
        plt.plot(history[i].history['loss'], label='train loss')
        plt.plot(history[i].history['acc'], label='train acc')
        plt.plot(history[i].history['val_loss'], label='val loss')
        plt.plot(history[i].history['val_acc'], label='val acc')
        plt.legend()
        if save:
            save_name = base_name + "_split%02d.png" % (i+1)
            plt.savefig(save_name, dpi=300)
        plt.show()

def save_predictions(model, history, model_metrics, model_predictions, name='results'):
    assert len(history)==len(model_metrics)==len(model_predictions)

    for i in range(len(history)):
        print(model.metrics_names, model_metrics[i])
        extension = '_split%02d.txt' % (i+1)
        with open(name+extension, 'w') as f:
            f.write(repr(model_predictions[i]))
            
def get_data(train_val_connection, test_connection, target_size, num_training_cases=-1, training_cases_multiplier=1, get_one_hot_labels=True):      
    # Get number of cases
    # Either set a number of training cases to use, or set how often each case should be used.
    # If both are not set, use each case once per epoch.
    if num_training_cases < 0:
        num_train_cases = train_val_connection.training_case_count() * training_cases_multiplier
    else:
        num_train_cases = num_training_cases
    num_val_cases = train_val_connection.validation_case_count()
    num_test_cases = test_connection.training_case_count()

    # Get all train/validation/test patches
    x, y = train_val_connection.get_training_batch(num_train_cases)
    images_train = np.array(x).reshape((num_train_cases,)+target_size)
    if target_size[-1] == 3: # Create color images
        images_train = np.pad(images_train, ((0,0),(0,0),(0,0),(1,1)), mode='symmetric')
    if get_one_hot_labels:
        labels_train = to_categorical(np.array([label.flatten() for label in y]))
    else:
        labels_train = np.array([label.flatten() for label in y])

    x, y = train_val_connection.get_validation_batch(num_val_cases)
    images_val = np.array(x).reshape((num_val_cases,)+target_size)
    if target_size[-1] == 3: # Create color images
        images_val = np.pad(images_val, ((0,0),(0,0),(0,0),(1,1)), mode='symmetric')
    if get_one_hot_labels:
        labels_val = to_categorical(np.array([label.flatten() for label in y]))
    else:
        labels_val = np.array([label.flatten() for label in y])
    
    x, y = test_connection.get_training_batch(num_test_cases)
    images_test = np.array(x).reshape((num_test_cases,)+target_size)
    if target_size[-1] == 3: # Create color images
        images_test = np.pad(images_test, ((0,0),(0,0),(0,0),(1,1)), mode='symmetric')
    if get_one_hot_labels:
        labels_test = to_categorical(np.array([label.flatten() for label in y]))
    else:
        labels_test = np.array([label.flatten() for label in y])

    return images_train, labels_train, images_val, labels_val, images_test, labels_test


def get_data_generators(paths_dict, batch_size=32, target_size=(109,91)):
    '''
    Returns generators for batches of train/val/test data collected from the folders given to the function. 
    
    Params: 
        paths_dict: Dictionary with three paths in 'train', 'val', 'test' pointing to folders with images to use. Subfolders are assumed to contain classes. Mandatory.
        batch_size: Batch size. Default=32
        target_size: 2D size tuple to rescale the images to. Default=(109,91)
    Returns:
        train_generator, val_generator, test_generator: Generator functions to retrieve a batch of data
    '''
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    test_datagen =  ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(paths_dict['train'], 
                                                 target_size=target_size,
                                                 color_mode='grayscale',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

    val_generator = train_datagen.flow_from_directory(paths_dict['val'],
                                                 target_size=target_size,
                                                 color_mode='grayscale',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

    test_generator = test_datagen.flow_from_directory(paths_dict['test'], 
                                                 target_size=target_size,
                                                 color_mode='grayscale',
                                                 batch_size=566,
                                                 class_mode='categorical',
                                                 shuffle=False) # don't shuffle so that the file names and indices are in sync.
    
    return train_generator, val_generator, test_generator