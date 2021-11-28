## Reviews classifiacator
### Data set
Open data set of reviwes from Netflix. THere a re 2 categories: positive (1) and negative (0). Data set was splited to test, valisation and train outside the project.

### Data processing
Keras Tokenizer was used for the processing. Output values were converted to one hot encoding

### Model
There is a simple model based on Embedding and Dense layers. Best accuracy on validation and test sets is 89%.

### Important note
This project was made as a part of the AI course. Model could be trained with the better accuracy.
