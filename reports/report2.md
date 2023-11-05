# Report on Final Solution Creation Process

## Introduction:
This report details the final solution I developed for a specific problem. After exploring a dictionary-based approach, I conducted research and ultimately implemented a seq2seq model using a tutorial on NLP from PyTorch.

1. Previous Approaches:

1.1. Dictionary-Based Solution:
Initially, I experimented with a basic dictionary-based approach. While this solution could handle simple cases, it lacked the flexibility and adaptability necessary for more complex problems.

1.2 Basic seq2seq Solution:
Before the introduction of attention mechanisms, the most common approach for sequence-to-sequence (seq2seq) models was to use a basic encoder-decoder architecture.

2. Research:

2.1. PyTorch NLP Tutorial:
To enhance my solution and gain a deeper understanding of natural language processing (NLP) techniques, I followed a tutorial on NLP from PyTorch. The tutorial served as a valuable resource, providing step-by-step guidance on implementing seq2seq models.

3. Seq2Seq Model with Attention:

3.1. Model Architecture:
Based on the tutorial, I implemented a seq2seq model architecture. The model consists of an encoder, which processes the input sequence, and a decoder, which generates the corresponding output sequence. This architecture is suitable for NLP tasks, particularly language generation, and can accommodate input sequences of varying lengths.


4. Training and Evaluation:

4.1. Training Process:
I trained the seq2seq model using the prepared dataset and the techniques outlined in the PyTorch NLP tutorial. This involved optimizing the model with backpropagation and utilizing an optimization algorithm such as Adam. Initially, I implemented training for both the model with attention and the model without attention.

4.2. Time and Performance Challenges:
During training, I observed that the model with attention required more time to converge compared to the model without attention. But, the model with attention showed better performance compared to the simpler model.

5. Refinement and Decision:

5.1. Iterative Evaluation:
Throughout the training and evaluation process, I monitored the performance of both models, experimenting with different hyperparameters and architectural adjustments.

5.2. Decision to Manipulate Dataset:
Based on my observations and reflections, I made the decision to reduce the dataset, I fixed the max length of tokenized sentence to 128 and also made sure that reference's toxicity is bigger than translation as it is supposed to be.

Conclusion:

In conclusion, I developed a final solution using a seq2seq model architecture. To manage to finish it on time I had to do some changes in datasets(reducing to approximately 60-70% of the initial size).