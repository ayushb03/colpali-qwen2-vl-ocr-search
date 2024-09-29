import torch
import streamlit as st
import logging
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import re

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app_debug.log"),  # Log to a file
        logging.StreamHandler() 
    ]
)

logging.info("Starting the application")

# Load models with caching
@st.cache_resource
def load_model_and_processor():
    logging.info("Loading model and processor")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.float32
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        logging.info("Model and processor loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model and processor: {e}")
        raise
    return model, processor

model, processor = load_model_and_processor()
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

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

def highlight_keyword(text: str, keyword: str):

    """Highlight all occurrences of the keyword in the given text using HTML."""

    highlighted_text = re.sub(
        f"({re.escape(keyword)})", r'<mark style="background-color: yellow">\1</mark>', text, flags=re.IGNORECASE
    )
    return highlighted_text

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

# Streamlit app starts here
st.title("OCR + Document Search: Extracting English & Hindi text from image")
logging.info("Streamlit app initialized")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    logging.info(f"Image uploaded: {uploaded_file.name}")

text_query = "Extract all the text in Sanskrit and English from the image."
keyword = st.text_input("Enter a keyword for search:")

if st.button("Process"):
    logging.info("Process button clicked")
    if uploaded_file and text_query and keyword:
        try:
            image_path = uploaded_file.name
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logging.info(f"Image saved to {image_path}")

            result = pipeline(image_path, text_query, keyword)

            st.subheader("JSON output")
            st.write(result)

            st.subheader("Search Results")
            if result["matched_sentences"]:
                for sentence in result["matched_sentences"]:
                    highlighted_sentence = highlight_keyword(sentence, keyword)
                    st.markdown(highlighted_sentence, unsafe_allow_html=True)
            else:
                st.info("No matches found for the keyword.")

            logging.info("Output displayed on Streamlit")
        except Exception as e:
            logging.error(f"Error processing the uploaded image: {e}")
            st.error(f"Error: {e}")
    else:
        logging.warning("Incomplete input - image, text query, or keyword missing")
        st.error("Please upload an image and fill in the text query and keyword.")
