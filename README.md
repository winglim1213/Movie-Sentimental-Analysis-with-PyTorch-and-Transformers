# Movie-Sentimental-Analysis-with-PyTorch-and-Transformers
## Set up 
### Dataset
In this project we will perform sentimental analysis on IMBD movie reviews. The dataset can be found in the following link http://ai.stanford.edu/~amaas/data/sentiment/ (Reference at the bottom of this page). The label of the dataset is binary with 1 for positive rating and 0 for negative rating. 
### Installation 
In this NLP task, We have used python 3.X and pytorch with version '1.7.1+cu101'. The model is run based on GPU based pytorch. If your computer does no GPU, device will be set to cpu such that the model will be trained with CPU. There could lead to a longer computational time but will not affect the accurary. 

We will use BERT (Bidirectional Encoder Representations from Transformers) provided in transformer library developed by huggingface using. We will utilize the pretrained model and add an additional final layer with activation function for generating the final output based on torch model. 

For vistualization of the result, we will generate a classification report provided by sklearn.
## Modeling 
This project is divided into different parts. Main parts includes Preprocessing, Building the nueral network model, Setting up training and evaluation iteration and Visulization of  result. For details please read the jupyter notebook.
### Preprocessing
Since the documents are provided individually, we combine the dataset into a list as a corpus for a better control. Then we have our preprocessing part where we remove html tags, non letters charaters, * and . and excessive whitespace since these characters provide no predictive power. One may try different preprocessing method and look for an empirally better set of method for this dataset. Removing punctuation could be unnecessary since punctuation can possibly help the model to understand the corpus better by comparing the nearby words and label. However punctuation could be sometime messy and provide wrong information and the amount of data we feed may be insufficient for learning punctuation. We therefore removed punctuation in this project. Besides, we have removed numbers since numbers are also unnecessary in evaluating the rating of a movie, while insufficient data related to numbers could also lead to a worse prediction. 

After cleaning the dataset, we will tokenize the corpus using bert tokenizer from pretrained library. We will retrieve the input_ids and attention_mask from the encoder since these are the required inputs for the bert model. We will then send the tokenized corpus to the dataloader which will be later used to load the data to the model.
### Building the nueral network model
As previously mentioned, we will build the last layer of the pretrained model for generating the necessary output on top of the pretrained model. We will first pass it to a dropout layer and followed by the last linear layer with output size of 2 due to the binary label. Before returning the output the result will pass through an activation layer. In the original paper they use softmax but in this project we choose sigmoid due to the binary label. 

### Setting up training and evaluation iteration
This part is similar to any other pytorch neural network model. There are many great explanation on the internet, so I will simply skip the details.
### Visulization of result
Our model has accuracy of 93% which is an excellent result given the small dataset. Overall number of false positive and false negative is acceptable with false positive slightly higher than the other.

## Conclusion and Discussion 
The final result is applauding with 93% overall accuracy. One may improve the accuracy with trying different preprocessing method when cleaning the dataset, adjust the train-val-test ratio to 70-15-15, use larger pretrained model etc. 


## Reference 
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
