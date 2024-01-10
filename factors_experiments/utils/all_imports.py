import sys
sys.path.append("/home/qjx0814/FastEdit")
sys.path.append("/home/qjx0814/EasyEdit")
sys.path.append("/home/qjx0814/Ripple_Effect_Analysis/gradient_experiment")
import torch
import random
from fastedit.utils.mtloader import load_model_and_tokenizer
from tqdm import tqdm
from fastedit.utils.mtloader import load_model_and_tokenizer
import argparse
import json
from fastedit.utils.generate import generate_fast
from fastedit.rome import ROMEHyperParams,apply_rome_to_model
from fastedit.utils.template import Template
import os
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
torch.cuda.set_device(4)
import seaborn as sns
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import copy
import openai
openai.api_key = "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"
from tqdm import tqdm
import io
from utils.util import *
from contextlib import redirect_stdout, redirect_stderr
