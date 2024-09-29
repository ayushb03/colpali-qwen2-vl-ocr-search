import logging
import base64
import json
import re 
import time
import tempfile

import torch
import streamlit as st
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel

# set up logging config
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app_debug.log"),  # Log to a file
        logging.StreamHandler()  # Also output logs to console
    ]
)

logging.info("Starting the application")

# load models with caching
@st.cache_resource
def load_model_and_processor():
    logging.info("Loading model and processor")
    try:
        rag = RAGMultiModalModel.from_pretrained("vidore/colpali")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.float32
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        logging.info("Model and processor loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model and processor: {e}")
        raise
    return rag, model, processor

rag, model, processor = load_model_and_processor()
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

def build_index(image_path):
    rag.index(
        input_path=image_path,
        index_name="image_index",  # index will be saved at index_root/index_name/
        store_collection_with_index=True,
        overwrite=True
    )

def query_index(text_query):
    results = rag.search(text_query, k=1)
    return results


def colpali_index_search(uploaded_file, text_query):
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_file.write(uploaded_file.getbuffer())  # Use getbuffer() to get the bytes of the uploaded file
        temp_file.flush()   
        
        build_index(temp_file.name)  
        results = query_index(text_query)  
        print(f"Results from ColPali search: {results}")
        
        # if results:
        #     top1_img = base64.b64decode(results[0].base64)  
        #     return top1_img  
        # else:
        #     return None 

        return uploaded_file

def keyword_search(output_texts: list, keyword: str):
    logging.debug(f"Searching for keyword '{keyword}' in the output texts")
    keyword_lower = keyword.lower()
    matched_sentences = []

    for output_text in output_texts:
        sentences = output_text.split('. ')
        for sentence in sentences:
            if keyword_lower in sentence.lower():
                matched_sentences.append(sentence)

    logging.debug(f"Matched sentences: {matched_sentences}")
    return matched_sentences

# Function to highlight keyword in sentences
def highlight_keyword(sentences, keyword):
    highlighted_sentences = []
    highlight_style = f"""
    <style>
        .highlight {{
            background-color: #ffcc00;  /* Bright yellow background */
            color: #000000;               /* Black text */
            font-weight: bold;            /* Bold text */
            padding: 2px 5px;            /* Padding around text */
            border-radius: 5px;          /* Rounded corners */
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.5); /* Subtle shadow */
        }}
    </style>
    """
    
    for sentence in sentences:
        # Use re.sub for case-insensitive replacement
        highlighted_sentence = re.sub(re.escape(keyword), 
                                       f"<span class='highlight'>{keyword}</span>", 
                                       sentence, 
                                       flags=re.IGNORECASE)
        highlighted_sentences.append(highlighted_sentence)
    
    return highlight_style + "".join(highlighted_sentences)


def pipeline(image_path: str, text_query: str, keyword: str):
    logging.info(f"Starting pipeline for image: {image_path}")
    try:
        image = Image.open(image_path)
        logging.debug("Image opened successfully")
    except Exception as e:
        logging.error(f"Error opening image: {e}")
        raise

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_query},
            ],
        }
    ]

    try:
        logging.info("Processing vision info")
        chat_template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[chat_template],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        logging.info("Generating output")
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        matched_sentences = keyword_search(output_text, keyword)

        response = {
            "text_query": text_query,
            "keyword": keyword,
            "output_text": output_text,
            "matched_sentences": matched_sentences
        }

        json_output = json.dumps(response, ensure_ascii=False, indent=4)
        logging.info("Pipeline completed successfully")
        return json_output  
    except Exception as e:
        logging.error(f"Error in the pipeline process: {e}")
        raise

######## streamlit app ########

st.title("OCR + Document Search: Extracting English & Hindi text from image")
logging.info("Streamlit app initialized")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    logging.info(f"Image uploaded: {uploaded_file.name}")


text_query = "Extract all the text in Sanskrit and English from the image."
indexed_image = None

if st.button("Build Index"):

    with st.spinner("Building the index Please wait...."):
        logging.info("Build Index button clicked")
        if uploaded_file:
            time.sleep(2)
            #indexed_image = colpali_index_search(uploaded_file, text_query)
        else:
            st.error("Please upload an image.")

keyword = st.text_input("Enter a keyword for search:", placeholder="Type your keyword here...")
indexed_image = uploaded_file

if st.button("Process"):
    logging.info("Process button clicked")
    if indexed_image and keyword:
        try:
            # Save the uploaded image
            image_path = indexed_image.name
            with open(image_path, "wb") as f:
                f.write(indexed_image.getbuffer())
            logging.info(f"Image saved to {image_path}")

            with st.spinner("Processing... Please wait."):
                time.sleep(2)  
                result = pipeline(image_path, text_query, keyword)

            result_dict = json.loads(result)
            st.subheader("JSON output")
            st.write(result_dict)
            
            # extract matched sentences from the JSON output
            matched_sentences = result_dict.get("matched_sentences", [])

            # display matched sentences with highlighted keyword
            st.subheader("Highlighted Sentences:")
            if matched_sentences:
                highlighted_content = highlight_keyword(matched_sentences, keyword)
                st.markdown(highlighted_content, unsafe_allow_html=True)
            else:
                st.write("No sentences matched the keyword.")

        except Exception as e:
            logging.error(f"Error processing the image: {e}")
            st.error("An error occurred while processing the image. Please try again.")

# Footer section
st.markdown("---")
st.write("Made with ❤️ by AYUSH BODADE")

# Links section
st.markdown(
    """
    ### Connect with me:
    - [GitHub](https://github.com/ayushb03)
    - [LinkedIn](https://www.linkedin.com/in/ayushbodade)
    - [Email](mailto:ayushbodade1@gmail.com)
    - [Twitter](https://twitter.com/ayushb03)
    """
)
