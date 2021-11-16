# Multiplex Model of Mental Lexicon with Visual and Word Embeddings

This project is divided into two phases:

*Phase 1*
- Create a three new network layers: word embeddings using GloVe, sensorimotor layer using the Lancaster dataset, and a visual layer using Words as Classifiers. Compare the new layers with the already existing multiplex network. Analyze the differences in information stored in either network. 

The following table, inspired by Table 1 from [this paper](https://github.com/FloCiaglia/multiplex_network/blob/main/papers/Multiplex_lexical_networks.pdf), illustrates the results from phase 1. 

![name](https://github.com/FloCiaglia/multiplex_network/blob/main/data/Evaluation_phase1.png)

Note: if a number in the table is in a different color, it means that it differs from what we were expecting. 

*Phase 2*
- Attach new word nodes to a network with small amount of initial words using preferential attachment. Compare the final result network with the one generated on the full vocabulary.   
	

### Libraries Used

- Python 3.7.6
- NetworkX 2.6
- Pytorch 1.9.1

 


