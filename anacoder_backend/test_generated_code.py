# Importing Required Libraries for Testing
import unittest
import os
import shutil
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from PIL import Image
from main import preprocess_images, build_model, train_and_evaluate

class TestImageClassifier(unittest.TestCase):

    def setUp(self):
        # Set up a temporary directory for the image dataset
        os.makedirs('./test_data/class1')
        os.makedirs('./test_data/class2')

        # Create dummy images
        img1 = Image.new('RGB', (200, 200), color = (73, 109, 137)) # Image for class1
        img2 = Image.new('RGB', (300, 300), color = (209, 200, 255)) # Image for class2

        self.img1_path = './test_data/class1/img1.jpg'
        self.img2_path = './test_data/class2/img2.jpg'

        img1.save(self.img1_path)
        img2.save(self.img2_path)

        # Prepare a trained model for testing
        self.model = build_model(input_shape=(128, 128, 3))
        
        # Create dummy training data
        self.X_train = np.random.rand(5, 128, 128, 3)
        self.y_train = utils.to_categorical(np.random.randint(2, size=(5, 1)), num_classes=2)
        
        # Create dummy testing data
        self.X_test = np.random.rand(2, 128, 128, 3)
        self.y_test = utils.to_categorical(np.random.randint(2, size=(2, 1)), num_classes=2)
        
    def tearDown(self):
        # Delete the temporary directory and contents after the test
        shutil.rmtree('./test_data')

    def test_preprocess_images(self):
        images, labels = preprocess_images('./test_data')
        self.assertIsNotNone(images)
        self.assertIsNotNone(labels)
        self.assertEqual(len(images), 2)
        self.assertEqual(len(labels), 2)
        self.assertEqual(images[0].shape, (128, 128, 3))

    def test_build_model(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.model.layers), 10)

    def test_train_and_evaluate(self):
        model = train_and_evaluate(self.model, self.X_train, self.y_train, self.X_test, self.y_test, batch_size=2, epochs=1)
        self.assertIsNotNone(model)
        self.assertTrue(os.path.isfile('trained_model.h5'))

    def test_load_trained_model(self):
        model = load_model('trained_model.h5')
        self.assertIsNotNone(model)
        

if __name__ == '__main__':
    unittest.main()