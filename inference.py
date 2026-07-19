"""
Inference Script for Farmer Crop Advisory AI
Usage: python inference.py --question "गेहूं में पीला रतुआ रोग के लक्षण क्या हैं?"
"""
import torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = """आप भारतीय किसानों के लिए एक कृषि विशेषज्ञ सहायक हैं।
आप फसल रोग, कीट प्रबंधन, उर्वरक और खेती की सलाह देते हैं।
हमेशा व्यावहारिक सलाह दें और स्थानीय KVK से पुष्टि करने की सलाह दें।"""


def load_model(model_id="me-nabi/farmer-advisory-hindi-qwen2.5-3b"):
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        ),
        device_map="auto", torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = PeftModel.from_pretrained(base_model, model_id)
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, question, max_tokens=200):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.7,
            do_sample=True, repetition_penalty=1.1,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="सरसों में माहू कीट के लिए कौन सा कीटनाशक छिड़कें?")
    parser.add_argument("--model_id", default="me-nabi/farmer-advisory-hindi-qwen2.5-3b")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.model_id)
    print(f"\nQ: {args.question}")
    print(f"A: {ask(model, tokenizer, args.question)}")