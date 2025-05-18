# Example of using the saved BERT model for inference on new text
# This code can be added to a new cell at the end of the notebook

from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
from scipy.special import softmax
import numpy as np

# Path to saved model
model_path = "./results_bert/best_model"

# Load model and tokenizer
inference_model = BertForSequenceClassification.from_pretrained(model_path)
inference_tokenizer = BertTokenizerFast.from_pretrained(model_path)

# Set model to evaluation mode
inference_model.eval()

# Function to predict sentiment of new text
def predict_sentiment(text, model, tokenizer):
    # Tokenize input text
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=256, return_tensors="pt")
    
    # Move to the same device as model
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model = model.to('cuda')
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probabilities
    probs = softmax(outputs.logits.cpu().numpy(), axis=1)
    
    # Get prediction and confidence
    prediction = np.argmax(probs, axis=1)[0]
    confidence = probs[0, prediction]
    
    # Map prediction to label
    sentiment = "positive" if prediction == 1 else "negative"
    
    return sentiment, confidence

# Example texts for testing
test_texts = [
    "This movie was absolutely amazing! The actors were brilliant and the plot was fantastic.",
    "What a terrible waste of time. The plot made no sense and the acting was wooden.",
    "It was okay, not great but not awful either. Some parts were good, others were boring.",
    "Despite some flaws in the screenplay, the film delivers a powerful message with stunning visuals."
]

print("Testing BERT model on new reviews:\n")
for text in test_texts:
    sentiment, confidence = predict_sentiment(text, inference_model, inference_tokenizer)
    print(f"Text: {text}")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")
    print("-" * 80)
