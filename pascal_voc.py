import os
import numpy as np

class pascal_voc(imdb):
  def __init__(self, image_set, year, use_diff=False):
    name = 'voc_'+year+'_'+image_set

