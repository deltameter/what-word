# What-Word

What-Word is a an implementation of a simple multilayer perceptron trained for the purpose recognizing whole hand-written English words. This was implemented with the knowledge I'd gained from Andrew Ng's great [Introduction to Machine Learning](https://www.coursera.org/learn/machine-learning) course on Coursera. 

### Pipeline
- When fed in an image of a whole word, it is split up into individual characters based off of whitespace delimiting.  
- The neural net preforms OCR on these individual characters.
- The characters from the neural net are then matched to a word in the English dictionary based off of their softmax probabilities.

### How this project reinforced my knowledge 
- Features choice matters. A lot. One of the methods of fixing high variance in the Andrew Ng course was to choose fewer features. I implemented this by stripping all of the whitespace from the NIST images before feeding them in. The CV accuracy immediately jumped ~7-8%.
- High variance is almost always ameliorated with more data. At first I trained with only one of the subsets of the dataset, but eventually when I used the full dataset, CV accuracy jumped ~3%. 
- GPUs are really, really good for ML. I trained on my (reasonably powerful) laptop with an integrated GPU and the model took 2 hours to converge. I trained on my desktop with a GTX 970 and it took several minutes. 

### Why not a Convolutional Neural Net?
I wanted to make this using only tools that I truly understood, and since I started this project immediately after completing the neural network portion of Andrew Ng's Intro to ML course, I hadn't been exposed to or truly learned about CNNs.


NN was trained with the [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19) 

