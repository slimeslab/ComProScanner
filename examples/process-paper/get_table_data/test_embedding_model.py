import torch
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")

# Test word pairs for comparison
test_pairs = [
    ("C6H12O6", "glucose"),
    ("C6H5OH", "phenol"),
    ("Zr(NO3)4·5H2O", "zirconium(IV) nitrate pentahydrate"),
    ("CVD", "chemical vapor deposition"),
    ("XRD", "X-ray diffraction"),
    ("SEM", "scanning electron microscopy"),
    ("PVDF", "polyvinylidene fluoride"),
    ("PVC", "polyvinyl chloride"),
    ("PMMA", "polymethyl methacrylate"),
    ("DOS", "density of states"),
    ("ESP", "electrostatic potential"),
    ("DFT", "density functional theory"),
]


def load_physbert_model():
    """Load PhysBERT model"""
    try:
        from transformers import AutoModel, AutoTokenizer

        physbert_model_name = "thellert/physbert_cased"
        tokenizer = AutoTokenizer.from_pretrained(physbert_model_name)
        model = AutoModel.from_pretrained(physbert_model_name)
        print("✓ PhysBERT model loaded successfully")
        return model, tokenizer, True
    except (ImportError, Exception) as e:
        print(f"✗ PhysBERT model not available: {e}")
        return None, None, False


def load_sentence_transformer():
    """Load sentence transformer model"""
    try:
        from sentence_transformers import SentenceTransformer

        model_name = "all-mpnet-base-v2"
        model = SentenceTransformer(model_name)
        print("✓ Sentence-transformer model loaded successfully")
        return model, True
    except ImportError as e:
        print(f"✗ Sentence-transformers not available: {e}")
        return None, False


def calculate_physbert_similarity(text1, text2, model, tokenizer):
    """Calculate similarity using PhysBERT"""
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Use CLS token embeddings for sentence representation
    embedding1 = outputs1.last_hidden_state[:, 0, :]
    embedding2 = outputs2.last_hidden_state[:, 0, :]

    # Normalize embeddings
    embedding1 = embedding1 / embedding1.norm(dim=1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(dim=1, keepdim=True)

    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        embedding1, embedding2, dim=1
    ).item()

    return similarity


def calculate_sentence_transformer_similarity(text1, text2, model):
    """Calculate similarity using sentence transformer"""
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    similarity = torch.nn.functional.cosine_similarity(
        embedding1.unsqueeze(0), embedding2.unsqueeze(0), dim=1
    ).item()

    return similarity


def main():
    print("Model Comparison: PhysBERT vs all-mpnet-base-v2")
    print("=" * 60)

    # Load models
    physbert_model, physbert_tokenizer, physbert_available = load_physbert_model()
    sentence_model, sentence_available = load_sentence_transformer()

    if not physbert_available and not sentence_available:
        print("No models available for comparison!")
        return

    print("\nTesting similarity scores on chemistry/materials science pairs...")
    print("=" * 60)

    # Prepare results for sorting
    all_results = []

    for text1, text2 in test_pairs:
        result_data = {"text1": text1, "text2": text2}

        # Calculate PhysBERT similarity
        if physbert_available:
            physbert_score = calculate_physbert_similarity(
                text1, text2, physbert_model, physbert_tokenizer
            )
            result_data["physbert_score"] = physbert_score
            result_data["physbert_display"] = f"{physbert_score:.4f}"
        else:
            result_data["physbert_score"] = None
            result_data["physbert_display"] = "N/A"

        # Calculate sentence transformer similarity
        if sentence_available:
            sentence_score = calculate_sentence_transformer_similarity(
                text1, text2, sentence_model
            )
            result_data["sentence_score"] = sentence_score
            result_data["sentence_display"] = f"{sentence_score:.4f}"
        else:
            result_data["sentence_score"] = None
            result_data["sentence_display"] = "N/A"

        # Calculate difference
        if physbert_available and sentence_available:
            difference = abs(physbert_score - sentence_score)
            result_data["difference"] = difference
            result_data["difference_display"] = f"{difference:.4f}"
        else:
            result_data["difference"] = float("inf")  # For sorting when N/A
            result_data["difference_display"] = "N/A"

        all_results.append(result_data)

    # Sort by difference (highest to lowest)
    all_results.sort(key=lambda x: x["difference"], reverse=True)

    # Prepare table data
    results = []
    headers = ["Text 1", "Text 2", "PhysBERT", "all-mpnet-base-v2", "Difference"]

    for result in all_results:
        row = [
            result["text1"],
            result["text2"],
            result["physbert_display"],
            result["sentence_display"],
            result["difference_display"],
        ]
        results.append(row)

    # Print results table (sorted by difference, high to low)
    print("\nComparison Results (sorted by difference - highest to lowest):")
    print(tabulate(results, headers=headers, tablefmt="grid", stralign="center"))


if __name__ == "__main__":
    main()
