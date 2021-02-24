# Movie-Sentimental-Analysis-with-PyTorch-and-Transformers

## Set up 

### Dataset
In this project we will perform sentimental analysis on IMBD movie reviews. The dataset can be found in the following link http://ai.stanford.edu/~amaas/data/sentiment/ (Reference at the bottom of this page).
### Installation 
In this NLP task, I have used python 3.X and pytorch with version '1.7.1+cu101'. The model is run based on GPU based pytorch. If your computer does no GPU, device will be set to cpu such that the model will be trained with CPU. There could lead to a longer computational time but will not affect the accurary. 

We will use BERT (Bidirectional Encoder Representations from Transformers) provided in transformer library developed by huggingface using. I will utilize the pretrained model and add an additional final layer with activation function for generating the final output based on torch model. 


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
