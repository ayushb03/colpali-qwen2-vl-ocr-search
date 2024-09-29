# 🖼️ OCR + Document Search: ColPali Architecture + Qwen2-VL model

Welcome to the **OCR + Document Search App**, a powerful tool for extracting and searching text from images! This app leverages advanced machine learning models to identify and process English and Hindi text, providing a seamless user experience for document analysis.

## 🌐 Deployment

The OCR + Document Search App is deployed on [Hugging Face Spaces](https://huggingface.co/spaces). You can access it online to test its functionality without any setup on your local machine!

## ⚠️ Caution

Please use **super small files** to test the application. The models can take a long time to load, build the index, and then query, which may lead to delays with larger files, also try using the deployed app only as running these models locally is not fesible due to low compute resources.

## 📋 Features

- **Image Upload**: Easily upload images in JPG, JPEG, or PNG formats.
- **Indexing**: Build an index using the ColPali model using Byaldi library to retrieve the most relevant pages to answer the text query.
- **Keyword Search**: Search for specific keywords within extracted text using the Qwen2-VL-2B-Instruct VLM for querying.
- **Highlighting**: Matched keywords in the output are highlighted for better UX.
- **User-friendly Interface**: Built with Streamlit for an intuitive user experience.

## 🚀 How It Works

1. **Image Upload**: 
   - Users can upload their image files through the web interface. The app supports JPG, JPEG, and PNG formats.

2. **Build Index**:
   - After uploading an image, users can click the **Build Index** button. This process extracts relevant information and builds an index with the ColPali architecture for quick retrieval.
   - The application utilizes a **RAGMultiModalModel** to index the image (in the byaldi library), ensuring efficient search capabilities.

3. **Keyword Input**:
   - Users can input a keyword into a designated text box to search for specific terms in the extracted text.

4. **Processing**:
   - Once the index is built and a keyword is provided, users can click the **Process** button.
   - The app processes the image using a pipeline that involves:
     - Opening the image and preparing the inputs.
     - Generating text outputs through the **Qwen2VLForConditionalGeneration** model.
     - Searching for the specified keyword within the output text.

5. **Results Display**:
   - The app returns a JSON output containing the extracted text and matched sentences for the provided keyword.
   - Matched sentences are displayed with highlighted keywords for easy identification.

6. **User Interaction**:
   - Users can view the highlighted sentences, and the app provides feedback on the processing status, ensuring a smooth experience.

## 📚 Installation

To run this app locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ayushb03/colpali-qwen2-vl-ocr.git 
   
3. **Create & Activate Environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate 
   
4. **Install Dependencies && Run the app**
  ```bash
   pip install -r requirements.txt && streamlit run app.py

  
