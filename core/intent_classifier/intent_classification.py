# --- Complete code for ONNX model inference (standalone) ---

# Import necessary libraries
import numpy as np
from transformers import AutoTokenizer, DistilBertTokenizer
import onnxruntime as ort
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import torch 
import os
import time
import json
import gc


CURRENT_DIR = os.path.dirname(__file__)

ONNX_MODEL_PATH = os.path.join(CURRENT_DIR,"../../jp-models/washingBERT/line_distilbert_multi_head_classifier_int8.onnx") # bert int8

# Load the tokenizer
licorp_tokenizer_path = os.path.join(CURRENT_DIR,"../../jp-models/washingBERT/line-distilbert-base-japanese")
tokenizer = AutoTokenizer.from_pretrained(licorp_tokenizer_path, trust_remote_code=True,local_files_only=True)

#Load the ONNX runtime session
# Attempt to use CUDA provider if available
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

INTENT = []
TYPES = []
SEC_TYPES = []

INTENT_ENCODER_PATH = os.path.join(CURRENT_DIR,"../../jp-models/washingBERT/intent_classes.json")
TYPES_ENCODER_PATH = os.path.join(CURRENT_DIR,"../../jp-models/washingBERT/types_classes.json")
SEC_TYPES_ENCODER_PATH = os.path.join(CURRENT_DIR,"../../jp-models/washingBERT/sec_types_classes.json")

# Create LabelEncoder instances and fit them with the loaded classes
with open(TYPES_ENCODER_PATH, 'r') as f:
    types_classes = json.load(f)
types_encoder = LabelEncoder()
types_encoder.classes_ = np.array(types_classes)

with open(INTENT_ENCODER_PATH, 'r') as f:
    intent_classes = json.load(f)
intent_encoder = LabelEncoder()
intent_encoder.classes_ = np.array(intent_classes)


with open(SEC_TYPES_ENCODER_PATH, 'r') as f:
    sec_types_classes = json.load(f)
sec_types_encoder = LabelEncoder()
sec_types_encoder.classes_ = np.array(sec_types_classes)
# --- Helper function to calculate softmax ---
def softmax(x):
    """Calculate softmax scores for a numpy array."""
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)



def predict_onnx(sentence: str, tokenizer = tokenizer, ort_session = ort_session, intent_enc=intent_encoder, types_enc= types_encoder, sec_types_enc= sec_types_encoder):
    """
    Runs inference on a single sentence using the ONNX model,
    calculates confidence scores, and returns a structured dictionary.
    """

    # 1. Tokenize the input sentence
    # Use return_tensors="np" directly to get numpy arrays
    start_time = time.time()
    
    tokenized = tokenizer(sentence, return_tensors="np", truncation=True, padding="max_length", max_length=64)
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    # 2. Prepare inputs for ONNX runtime
    ort_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    # 3. Run inference
    ort_intent_logits, ort_types_logits, ort_sec_types_logits = ort_session.run(None, ort_inputs)

    # 4. Calculate probabilities using softmax
    intent_probs = softmax(ort_intent_logits)
    types_probs = softmax(ort_types_logits)
    sec_types_probs = softmax(ort_sec_types_logits)

    # 5. Get predicted index (argmax)
    pred_intent_idx = np.argmax(intent_probs, axis=1)[0]
    pred_types_idx = np.argmax(types_probs, axis=1)[0]
    pred_sec_types_idx = np.argmax(sec_types_probs, axis=1)[0]
    # 6. Get confidence score for the *predicted* class
    intent_confidence = float(intent_probs[0, pred_intent_idx])
    types_confidence = float(types_probs[0, pred_types_idx])
    sec_types_confidence = float(sec_types_probs[0, pred_sec_types_idx])

    # 7. Convert numerical predictions back to human-readable labels
    predicted_intent = intent_enc.inverse_transform([pred_intent_idx])[0]
    predicted_types = types_enc.inverse_transform([pred_types_idx])[0]
    predicted_sec_types = sec_types_enc.inverse_transform([pred_sec_types_idx])[0]
    latency_ms = (time.time() - start_time) * 1000
    # 8. Create the result dictionary and return it
    result = {
        "predicted_intent": predicted_intent,
        "intent_confidence": intent_confidence,
        "predicted_type": predicted_types,
        "type_confidence": types_confidence,
        "predicted_second_type": predicted_sec_types,
        "second_type_confidence": sec_types_confidence,
        "latency_ms":latency_ms
    }

    return result
