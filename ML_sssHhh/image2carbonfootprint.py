import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
import json

# Ensure PyTorch is installed
try:
    import torch
except ImportError:
    raise ImportError("Please install PyTorch by following the instructions at https://pytorch.org/get-started/locally/")

# Configure Gemini API
GOOGLE_API_KEY = 'AIzaSyAqfzrYDqh7lRvaX7YIZrCAosyMerJmHnY'  # Replace with your API key
genai.configure(api_key=GOOGLE_API_KEY)

class HARCarbonFootprint:
    def __init__(self, base_path):
        self.base_path = base_path
        self.image_size = (224, 224)
        
        # Setup paths
        self.test_dir = os.path.join(base_path, 'test')
        self.test_csv = os.path.join(base_path, 'Testing_set.csv')
        
        # Load pre-trained MobileNetV2 model
        self.model = MobileNetV2(weights='imagenet')
        
        # Initialize Gemini
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Initialize image captioning model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
    def generate_detailed_report(self, activity, predicted_class, confidence, caption):
        """Generate a detailed report for a given activity using the Gemini API"""
        try:
            prompt = (
                f"Generate a detailed report in JSON format explaining the data and predictions of the model. "
                f"Include information about the predicted class, confidence, and image caption. "
                f"Activity: {activity}, Predicted Class: {predicted_class}, Confidence: {confidence}, Caption: {caption}"
            )
            response = self.gemini_model.generate_text(
                prompt=prompt,
                max_tokens=150
            )
            detailed_report = response['choices'][0]['text'].strip()
            return json.loads(detailed_report)
        except Exception as e:
            print(f"Error generating detailed report for {activity}: {str(e)}")
            return {"error": str(e)}
    
    def predict_image(self, img_path):
        """Predict the class of a single image using MobileNetV2"""
        img = image.load_img(img_path, target_size=self.image_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        preds = self.model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=1)[0][0]
        predicted_class = decoded_preds[1]
        confidence = float(decoded_preds[2])
        
        return predicted_class, confidence
    
    def generate_caption(self, img_path):
        """Generate a caption for a single image using BLIP"""
        img = image.load_img(img_path, target_size=self.image_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        inputs = self.processor.images=img_array, return_tensors="pt")
        out = self.caption_model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def predict_test_set(self):
        """Predict on test set and analyze carbon footprint"""
        test_df = pd.read_csv(self.test_csv)
        predictions = []
        detailed_reports = []
        
        print("Processing test images...")
        for idx, row in test_df.iterrows():
            img_path = os.path.join(self.test_dir, row['filename'])
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                continue
                
            try:
                predicted_class, confidence = self.predict_image(img_path)
                caption = self.generate_caption(img_path)
                detailed_report = self.generate_detailed_report(row['filename'], predicted_class, confidence, caption)
                
                predictions.append({
                    'filename': row['filename'],
                    'Predicted_Class': predicted_class,
                    'Confidence': confidence,
                    'Caption': caption
                })
                
                detailed_reports.append({
                    'filename': row['filename'],
                    'Detailed_Report': detailed_report
                })
                
            except Exception as e:
                print(f"Error processing image {row['filename']}: {str(e)}")
                continue
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images")
        
        predictions_df = pd.DataFrame(predictions)
        detailed_reports_df = pd.DataFrame(detailed_reports)
        
        # Save results
        predictions_df.to_csv('predictions.csv', index=False)
        detailed_reports_df.to_csv('detailed_reports.csv', index=False)
        
        return predictions_df, detailed_reports_df

# Example usage
if __name__ == "__main__":
    BASE_PATH = r'D:\Technovate\New folder\Human Action Recognition'
    
    har_carbon = HARCarbonFootprint(BASE_PATH)
    predictions_df, detailed_reports_df = har_carbon.predict_test_set()
    
    print("\nPredictions Sample:")
    print(predictions_df.head())
    print("\nDetailed Reports Sample:")
    print(detailed_reports_df.head())