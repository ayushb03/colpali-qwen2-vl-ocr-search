# 🖼️ OCR + Document Search App

Welcome to the **OCR + Document Search App**, a powerful tool for extracting and searching text from images! This app leverages advanced machine learning models to identify and process English and Hindi text, providing a seamless user experience for document analysis.

## 📋 Features

- **Image Upload**: Easily upload images in JPG, JPEG, or PNG formats.
- **Indexing**: Build an index from uploaded images for quick searches.
- **Keyword Search**: Search for specific keywords within extracted text.
- **Highlighting**: Matched keywords in the output are highlighted for better visibility.
- **User-friendly Interface**: Built with Streamlit for an intuitive user experience.

## 🚀 How It Works

1. **Image Upload**: 
   - Users can upload their image files through the web interface. The app supports JPG, JPEG, and PNG formats.

2. **Build Index**:
   - After uploading an image, users can click the **Build Index** button. This process extracts relevant information and builds an index for quick retrieval.
   - The application utilizes a **RAGMultiModalModel** to index the image, ensuring efficient search capabilities.

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
   git clone <repository-url>
   cd <repository-directory>
