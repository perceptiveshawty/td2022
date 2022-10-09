# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
# from keras.models import load_model
# from tensorflow.keras.utils import load_img, img_to_array
import torch
from torchvision import models, transforms, datasets
from torchvision.io import read_image
from collections import OrderedDict

# Import helper functions from utils.py
import utils

class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def __init__(self):
        """
        Initializes any variables to be used when making predictions
        """
        # self.model = load_model('example_model.h5')
        model = models.resnet50(weights=None)

        # original saved file with DataParallel
        state_dict = torch.load('unpuzzle.pt', map_location='cpu')

        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()

        # dataset = datasets.ImageFolder('../PuzzleNet/train')
        # print("Mapping from class labels to digit strings:", dataset.class_to_idx)

        class2idx = {'0123': 0, '0132': 1, '0213': 2, '0231': 3, '0312': 4, '0321': 5, '1023': 6, '1032': 7, '1203': 8, '1230': 9, '1302': 10, '1320': 11, '2013': 12, '2031': 13, '2103': 14, '2130': 15, '2301': 16, '2310': 17, '3012': 18, '3021': 19, '3102': 20, '3120': 21, '3201': 22, '3210': 23}

        self.model = model
        self.T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])
        self.idx2class = {v: k for k, v in class2idx.items()}

    # def find_classes():
    #     classes = os.listdir('../PuzzleNetSplits/train')
    #     classes.sort()
    #     class_to_idx = {classes[i]: i for i in range(len(classes))}
    #     return classes, class_to_idx

    def make_prediction(self, img_path):
        """
        DO NOT RENAME THIS FUNCTION
        This function enables automated judging
        This function should stay named as `make_prediction(self, img_path)`

        INPUT:
            img_path: 
                A string representing the path to an RGB image with dimensions 128x128
                example: `example_images/1.png`
        
        OUTPUT:
            A 4-character string representing how to re-arrange the input image to solve the puzzle
            example: `3120`
        """



        # Load the image
        img = read_image(img_path)

        # Preprocess
        batch = self.T(img).unsqueeze(0)

        # Converts the image to a 3D numpy array (128x128x3)
        # img_array = img_to_array(img)

        # Convert from (128x128x3) to (Nonex128x128x3), for tensorflow
        # img_tensor = np.expand_dims(img_array, axis=0)

        # Preform a prediction on this image using a pre-trained model (you should make your own model :))
        prediction = self.model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        
        # category_name = weights.meta["categories"][class_id]
        category_name = self.idx2class[class_id]
        print(f"{category_name}: {100 * score:.1f}%")
        return category_name

        # The example model was trained to return the percent chance that the input image is scrambled using 
        # each one of the 24 possible permutations for a 2x2 puzzle
        # combs = [''.join(str(x) for x in comb) for comb in list(permutations(range(0, 4)))]

        # Return the combination that the example model thinks is the solution to this puzzle
        # Example return value: `3120`
        # return combs[np.argmax(prediction)]

# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':

    for img_name in glob('example_images/*'):
        # Open an example image using the PIL library
        example_image = Image.open(img_name)

        # Use instance of the Predictor class to predict the correct order of the current example image
        predictor = Predictor()
        prediction = predictor.make_prediction(img_name)
        # Example images are all shuffled in the "3120" order
        # print(prediction)

        # Visualize the image
        pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
        # Example images are all shuffled in the "3120" order
        final_image = Image.fromarray(np.vstack((np.hstack((pieces[3],pieces[1])),np.hstack((pieces[2],pieces[0])))))
        # final_image.show()