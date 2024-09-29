# 🖼️ OCR + Document Search: ColPali Architecture + Qwen2-VL 

Welcome to the **OCR + Document Search App**, a powerful tool for extracting and searching text from images! This app leverages advanced machine learning models to identify and process English and Hindi text, providing a seamless user experience for document analysis.

## 🌐 Deployment

The OCR + Document Search App is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/ayushb03/colpali-qwen2-ocr-search). You can access it online to test its functionality without any setup on your local machine!

## ⚠️ Caution

Please use **super small files** to test the application. The models can take a long time to load, build the index, and then query, which may lead to delays with larger files, also try using the deployed app only as running these models locally is not fesible due to low compute resources. And open the app in only 1 tab of browser. This is a prototype so the optimization features like: quanti

## Optimizations that can be implemented

# OCR and Document Search Optimization Techniques

This document outlines various techniques to optimize OCR (Optical Character Recognition) and document search applications, improving performance, efficiency, and user experience.

## Optimization Techniques (only some of them have been implemented in the current version)

### 1. Quantization
- **Description**: Reduce model weight precision from floating-point to lower precision (e.g., int8).
- **Benefit**: Decreases model size and speeds up inference with minimal accuracy loss.

### 2. Model Pruning
- **Description**: Remove weights or neurons that contribute little to model performance.
- **Benefit**: Reduces model size and improves inference speed.

### 3. Knowledge Distillation
- **Description**: Train a smaller model (student) to mimic a larger model (teacher).
- **Benefit**: Achieve similar performance with reduced computational requirements.

### 4. Image Preprocessing
- **Description**: Enhance image quality via resizing, denoising, and contrast adjustment.
- **Benefit**: Improves OCR accuracy and reduces processing time.

### 5. Batch Processing
- **Description**: Process multiple images in parallel.
- **Benefit**: Increases throughput and reduces overall processing time.

### 6. Efficient Data Loading
- **Description**: Use optimized file formats and lazy loading.
- **Benefit**: Faster image loading for quicker application responses.

### 7. Hardware Acceleration
- **Description**: Leverage GPUs or TPUs for model inference.
- **Benefit**: Significantly boosts performance for complex computations.

### 8. Asynchronous Processing
- **Description**: Implement asynchronous tasks for non-immediate feedback tasks.
- **Benefit**: Enhances app responsiveness and user experience.

### 9. Caching
- **Description**: Store results of processed images or searches.
- **Benefit**: Reduces processing time for repeated queries.

### 10. Use Efficient Models
- **Description**: Utilize lightweight architectures (e.g., MobileNet, EfficientNet).
- **Benefit**: Faster inference and lower resource usage.

### 11. Text Post-Processing
- **Description**: Apply algorithms for correcting OCR output.
- **Benefit**: Enhances the quality of extracted text and improves search accuracy.

### 12. Reduce Input Size
- **Description**: Limit the image size or resolution before processing.
- **Benefit**: Speeds up inference by reducing data volume.

### 13. Multi-threading
- **Description**: Use multiple threads for simultaneous task execution.
- **Benefit**: Improves application responsiveness, especially during I/O operations.

### 14. Monitoring and Profiling
- **Description**: Use tools to monitor performance and identify bottlenecks.
- **Benefit**: Enables informed optimizations and continuous improvement.

## Conclusion
Implementing these optimization strategies can significantly enhance the performance and user experience of OCR and document search applications. Choose techniques based on your specific use case and resource availability.


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

  
