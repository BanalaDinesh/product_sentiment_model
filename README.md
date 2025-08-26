# Airline Sentiment Classifier 🚀

This repository contains a fine-tuned Transformer model for **sentiment analysis** of airline-related tweets.  
The model predicts one of three classes:  
- **Negative** (red)  
- **Neutral** (blue)  
- **Positive** (green)  

The app uses **Gradio** for an interactive web interface.

---

## 📂 Files Included
- `airline_model/` – Folder containing the trained model (config.json, tokenizer files, and model weights).  
- `inference_colab.py` – Script to load and run the model in Colab.  
 

---

## 🛠️ How to Run in Google Colab

1. **Open Google Colab**  
   Go to [https://colab.research.google.com](https://colab.research.google.com) and create a new notebook.

2. **Mount Google Drive & Load Model**  
   Save the `airline_model` folder in your Google Drive (e.g., inside `MyDrive/tasks/`).  

   Then paste the following code cell into Colab and run it:

   ```python
   from google.colab import drive
   import torch
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   import gradio as gr

   # 1. Mount Google Drive
   drive.flush_and_unmount()  # Unmount previous mount (if any)
   drive.mount('/content/drive2')  # Fresh mount

   # 2. Load Model & Tokenizer
   MODEL_PATH = "/content/drive2/MyDrive/tasks/airline_model" #Model path from drive 
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
   model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   model.eval()

   id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
   label_colors = {0: "red", 1: "blue", 2: "green"}

   # 3. Prediction Function
   def predict(tweet):
       inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=64)
       inputs = {k: v.to(device) for k, v in inputs.items()}
       with torch.no_grad():
           outputs = model(**inputs)
           probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

       pred_label = torch.argmax(probs).item()
       confidence_dict = {id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))}

       # Build HTML output with color-coded predicted label
       html_output = f"<h3>Predicted Label: <span style='color:{label_colors[pred_label]}'>{id2label[pred_label]}</span></h3>"
       html_output += "<h4>Confidence Scores:</h4><ul>"
       for i, (lbl, conf) in enumerate(confidence_dict.items()):
           color = label_colors[i] if i == pred_label else "gray"
           html_output += f"<li><b>{lbl}:</b> <span style='color:{color}'>{conf}</span></li>"
       html_output += "</ul>"

       return html_output

   # 4. Launch Gradio Interface
   gr.Interface(
       fn=predict,
       inputs=gr.Textbox(lines=3, placeholder="Enter your tweet here..."),
       outputs=gr.HTML(),
       title="Airline Sentiment Classifier",
       description="Enter an airline-related tweet to see the predicted sentiment with color-coded confidence scores.",
       theme="default"
   ).launch(share=True)
