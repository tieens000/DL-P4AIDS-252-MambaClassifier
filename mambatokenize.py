from transformers import AutoModel, AutoTokenizer
import re
import underthesea

def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

# Hàm chuẩn hoá câu
def standardize_data(examples):
    text = examples["text"]

    # Xóa dấu ở cuối câu
    text = re.sub(r"[\.,\?]+$", "", text)

    # Xóa punctuation
    text = (
        text.replace(",", " ")
        .replace(".", " ")
        .replace(";", " ")
        .replace("“", " ")
        .replace(":", " ")
        .replace("”", " ")
        .replace('"', " ")
        .replace("'", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace("-", " ")
    )

    # Xóa multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Normalize
    text = text.strip().lower()

    examples["text"] = text
    return examples

def word_segment(examples):
    examples["text"] = underthesea.word_tokenize(examples["text"], format="text")
    return examples

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize dữ liệu
    Args:
        examples: Dữ liệu cần tokenize
        tokenizer: Tokenizer
        max_length: Chiều dài tối đa của token
    Returns:
        Dict ``input_ids``, ``labels``.
    """
    return {
        **tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ),
        "labels": examples["sentiment"],
    }

data = [
    {"text": "gậy ông đập lưng ông.", "sentiment": "positive"},
    {"text": "bài tập đa dạng , cụ thể .", "sentiment": "positive"},
    {"text": "thầy rất tận tâm nhiệt tình , quan tâm đến sinh viên , giảng bài dễ hiểu , kiến thức sâu rộng am hiểu cao.", "sentiment": "negative"},
]

phobert, tokenizer = load_bert()
segmented_data = list(map(lambda x: word_segment(standardize_data(x)), data))
print(segmented_data)

