# -*- coding: utf-8 -*-
"""
Kyle Cloud
January 28, 2020
Ensemble Method:
Sum the probability distributions
outputted by softmax.
"""

import numpy as np

class EnsembleModel():
    
    def __init__(self, models, classes): 
        self.__models = models # a list of trained models
        self.classes = classes
        
    def model_count(self): # counts the number of trained models in the list
        count = 0
        for model in self.__models:
            count += 1
        return count
        
    def add_model(self, model): # add a model to the list
        self.__models.append(model)
        
    def classify(self, image, num2fruit): # takes in an image and a dictionary
        
        sums = []
        
        for i in range(self.classes):
            sums.append(0.0)
        
        for model in self.__models:
            prob_dist = model.predict(image) # softmax outputs a list
            # containing the probabilities of an image belonging to each class
            for i in range(len(sums)): # sum the probablities
                # outputed by each model
                sums[i] += prob_dist[i]

        # the num2fruit dictionary maps the indices of the prob dist lists
        # to the fruit classes they represent
        return num2fruit[np.argmax(sums)]

    
    def confidence(self, image):
        
        sums = []
        
        for i in range(self.classes):
            sums.append(0.0)
        
        for model in self.__models:
            prob_dist = model.predict(image)
            
            for i in range(len(sums)):
                sums[i] = prob_dist[i]
            
            conf = np.argmax(sums) / self.model_count()
            return conf