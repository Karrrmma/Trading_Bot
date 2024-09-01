from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 
import torch.nn.functional as F
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)
        #The attention mask helps the model to ignore these padding tokens during the attention mechanism calculations.

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        #Softmax function, often used in the output layer of a classification 
        #network to normalize the outputs into probabilities.

        # Apply the softmax function to the summed logits 
        #across the batch dimension (0).
        # This converts the logits to probabilities, making them 
        #interpretable as the likelihood of each class.
        # 'dim=-1' indicates that the softmax is applied 
        #along the last dimension (the class dimension).
        result = F.softmax(torch.sum(result, 0), dim=-1)

        # Find the index of the maximum value in the 'result' 
        #tensor, which corresponds to the class with the highest probability.
        # Extract the probability of the predicted class using this index.
        
        max_index = torch.argmax(result)# Finding the index of the highest probability


        probability = result[max_index]# Extracting the highest probability

        # Map the index of the predicted class to the corresponding
        # sentiment label using the 'labels' list.

        sentiment = labels[max_index]
        return probability, sentiment
    else:
        return 0, labels[-1]


if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(['markets responded good to the news!','traders were good!'])
    print(tensor, sentiment)
    print(torch.cuda.is_available())