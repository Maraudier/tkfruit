# -*- coding: utf-8 -*-
"""
Kyle Cloud
January 17, 2020
* Updated January 19, 2020 - add model functionality

Prerequisite: Train 7 models.

The purpose of this class is to
have 7 trained models vote on how
to classify an image of fruit.
"""

import numpy as np
from statistics import mode

class VoteModel():
    
    def __init__(self, models): # take in 7 models
        self.__models = models
        
    # added on January 19, 2020
    def add_model(self, model):
        self.__models.append(model)
        
    def classify(self, image, num2fruit):
        votes = []
        for model in self.__models:
            prob_dist = model.predict(image) # softmax outputs a list
            vote = np.argmax(prob_dist) # index of max probability
            votes.append(vote)
        return num2fruit[mode(votes)]
        # still need a dictionary that maps the softmax indices to the fruits
    
    def confidence(self, image):
        votes = []
        for model in self.__models:
            prob_dist = model.predict(image)
            vote = np.argmax(prob_dist)
            votes.append(vote)
            
            choice_votes = votes.count(mode(votes))
            conf = choice_votes / len(votes)
            return conf
