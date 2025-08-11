from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import torch
import re
import hashlib

# ✅ Ensure Hugging Face cache directory is writable
os.environ["HF_HOME"] = "/data/huggingface"

app = Flask(__name__, template_folder="templates")

# ✅ Path to fine-tuned model
MODEL_DIR = "./finetuned-flan-t5"

# ✅ Load tokenizer and model at startup
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()
    model_loaded = True
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model_loaded = False

# ✅ Response cache for consistency
response_cache = {}

def generate_improved_suggestion(prompt, prompt_lower, clarity_score, completeness_score):
    """Generate more specific and helpful suggested prompts"""
    if clarity_score >= 4 and completeness_score >= 4:
        return prompt

    action_words, tech_keywords, purpose_keywords = [], [], []

    for word in ["create", "design", "write", "build", "develop"]:
        if word in prompt_lower:
            action_words.append(word)

    tech_map = {
        "google sheets": "Google Sheets",
        "excel": "Excel",
        "android": "Android",
        "python": "Python",
        "javascript": "JavaScript",
        "js": "JavaScript",
        "api": "API"
    }
    for key, val in tech_map.items():
        if key in prompt_lower:
            tech_keywords.append(val)

    purpose_map = {
        "track": "tracking",
        "tracking": "tracking",
        "alert": "notifications",
        "notify": "notifications",
        "notification": "notifications",
        "expense": "expense management",
        "budget": "expense management",
        "attendance": "attendance management",
        "assignment": "assignment tracking",
        "deadline": "deadline management"
    }
    for key, val in purpose_map.items():
        if key in prompt_lower:
            purpose_keywords.append(val)

    # Specific suggestion rules
    if "google sheets" in prompt_lower and "formula" in prompt_lower:
        if "expense" in prompt_lower and "alert" in prompt_lower:
            return ("Create a comprehensive Google Sheets formula system that tracks monthly expenses "
                    "by category, calculates running totals, and highlights overspending with conditional formatting.")
        return "Create a detailed Google Sheets formula that [specific function] with clear instructions."

    if "android" in prompt_lower:
        if "student" in prompt_lower and "attendance" in prompt_lower:
            return ("Design a complete Android student app with attendance tracking (QR codes), "
                    "assignment deadlines, grade tracking, and push notifications.")
        return "Design a detailed Android app concept with UI mockups, features, and technical requirements."

    if "python" in prompt_lower:
        if "api" in prompt_lower:
            return ("Write a complete Python script integrating with [specific API], including "
                    "authentication, error handling, and usage examples.")
        return "Write a Python script that [specific functionality] with documentation and error handling."

    if "write something" in prompt_lower or len(prompt) < 20:
        return ("Be more specific: describe purpose, audience, required features, and any technical "
                "constraints or preferences.")

    if clarity_score < 4 or completeness_score < 4:
        enhanced = prompt
        if not any(w in prompt_lower for w in ["specific", "detailed", "comprehensive"]):
            enhanced = enhanced.replace("Create", "Create a detailed").replace("Design", "Design a comprehensive").replace("Write", "Write a complete")
        if not any(w in prompt_lower for w in ["include", "with", "that"]):
            enhanced += " with examples and step-by-step instructions."
        return enhanced

    return prompt

def create_fallback_response(prompt, context):
    """Generate fallback response if model fails"""
    prompt_lower = prompt.lower()

    clarity_score = 3
    completeness_score = 3
    alignment_score = 3

    if len(prompt) < 20:
        clarity_score = completeness_score = 1
    elif len(prompt) > 100:
        clarity_score = completeness_score = 4

    if "write" in prompt_lower and "something" in prompt_lower:
        clarity_score = completeness_score = 1
    if "clear" in prompt_lower and "detailed" in prompt_lower:
        clarity_score = completeness_score = 4

    if context:
        if "professional" in context.lower():
            alignment_score = 4
        elif "friendly" in context.lower():
            alignment_score = 4

    clarity_comment = "Clear and specific." if clarity_score >= 3 else "Could be clearer."
    completeness_comment = "Detailed enough." if completeness_score >= 3 else "Lacks context."
    alignment_comment = "Fully aligned." if alignment_score >= 4 else "Needs tone adjustment."

    suggested = generate_improved_suggestion(prompt, prompt_lower, clarity_score, completeness_score)

    return (f"Scores: clarity={clarity_score}, completeness={completeness_score}, alignment={alignment_score}\n"
            f"Comments: clarity={clarity_comment}, completeness={completeness_comment}, alignment={alignment_comment}\n"
            f"Suggested: {suggested}")

def is_valid_model_response(response):
    """Check if model output matches expected format"""
    if not response or len(response.strip()) < 20:
        return False
    return all(key in response for key in ["Scores:", "Comments:", "Suggested:"])

def get_cache_key(prompt, context):
    return hashlib.md5(f"{prompt}|{context}".encode()).hexdigest()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        user_prompt = data.get("prompt", "").strip()
        context = data.get("context", "").strip()

        if not user_prompt:
            return jsonify({"error": "Please provide a prompt"}), 400

        cache_key = get_cache_key(user_prompt, context)
        if cache_key in response_cache:
            return jsonify({"response": response_cache[cache_key]})

        if model_loaded:
            try:
                full_prompt = f"""
You are an AI Prompt Auditor. 
Your job is to evaluate the given prompt and return a structured analysis.

Follow this exact output format, with no extra text before or after:

Scores: clarity=<1-5>, completeness=<1-5>, alignment=<1-5>
Comments: clarity=<short comment>, completeness=<short comment>, alignment=<short comment>
Suggested: <improved prompt>

Example:
Scores: clarity=4, completeness=3, alignment=5
Comments: clarity=Clear and concise., completeness=Could include more context., alignment=Tone matches well.
Suggested: Write a 500-word blog post on healthy breakfast ideas with nutritional information and recipes.

Now evaluate:
Prompt: {prompt}
Context: {context}
"""

                inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=256)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.85,
                        top_k=30,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        length_penalty=1.0,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )

                result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                if full_prompt in result:
                    result = result.replace(full_prompt, "").strip()

                if is_valid_model_response(result):
                    response_cache[cache_key] = result
                    if len(response_cache) > 1000:
                        for k in list(response_cache.keys())[:100]:
                            del response_cache[k]
                    return jsonify({"response": result})
                else:
                    print(f"⚠️ Invalid model output — using fallback. Output:\n{result}")

            except Exception as e:
                print(f"❌ Model generation failed: {e}")

        fallback = create_fallback_response(user_prompt, context)
        response_cache[cache_key] = fallback
        return jsonify({"response": fallback})

    except Exception as e:
        print(f"❌ Error in /generate: {e}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    response_cache.clear()
    return jsonify({"message": "Cache cleared successfully"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
