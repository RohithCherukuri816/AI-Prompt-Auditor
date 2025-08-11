from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import torch
import re
import hashlib
from collections import defaultdict

# ✅ Make sure Hugging Face cache is writable
os.environ["HF_HOME"] = "/data/huggingface"

app = Flask(__name__, template_folder="templates")

# ✅ Path to your fine-tuned model
MODEL_DIR = "./finetuned-flan-t5"

# ✅ Load tokenizer and model once at startup
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()
    model_loaded = True
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model_loaded = False

# Simple response cache to maintain consistency
response_cache = {}

def create_fallback_response(prompt, context):
    """Create a fallback response when model fails"""
    prompt_lower = prompt.lower()
    
    # Basic scoring logic based on prompt characteristics
    clarity_score = 3
    completeness_score = 3
    alignment_score = 3
    
    # Length-based scoring
    if len(prompt) < 20:
        clarity_score = 1
        completeness_score = 1
    elif len(prompt) > 100:
        clarity_score = 4
        completeness_score = 4
    
    # Content-based scoring
    if "write" in prompt_lower and "something" in prompt_lower:
        clarity_score = 1
        completeness_score = 1
    
    if "clear" in prompt_lower and "detailed" in prompt_lower:
        clarity_score = 4
        completeness_score = 4
    
    # Context-aware scoring
    if context and "professional" in context.lower():
        alignment_score = 4
    elif context and "friendly" in context.lower():
        alignment_score = 4
    
    # Comments
    clarity_comment = "Clear and specific." if clarity_score >= 3 else "Could be clearer."
    completeness_comment = "Detailed enough." if completeness_score >= 3 else "Lacks context."
    alignment_comment = "Fully aligned." if alignment_score >= 4 else "Needs tone adjustment."
    
    # Improved suggested prompt generation
    suggested = generate_improved_suggestion(prompt, prompt_lower, clarity_score, completeness_score)
    
    return f"Scores: clarity={clarity_score}, completeness={completeness_score}, alignment={alignment_score}\nComments: clarity={clarity_comment}, completeness={completeness_comment}, alignment={alignment_comment}\nSuggested: {suggested}"

def generate_improved_suggestion(prompt, prompt_lower, clarity_score, completeness_score):
    """Generate more specific and helpful suggested prompts"""
    
    # If the prompt is already good, suggest minor improvements
    if clarity_score >= 4 and completeness_score >= 4:
        return prompt  # Keep the original if it's already well-structured
    
    # Extract key elements from the prompt
    action_words = []
    if "create" in prompt_lower:
        action_words.append("create")
    if "design" in prompt_lower:
        action_words.append("design")
    if "write" in prompt_lower:
        action_words.append("write")
    if "build" in prompt_lower:
        action_words.append("build")
    if "develop" in prompt_lower:
        action_words.append("develop")
    
    # Extract technology/tools mentioned
    tech_keywords = []
    if "google sheets" in prompt_lower or "sheets" in prompt_lower:
        tech_keywords.append("Google Sheets")
    if "excel" in prompt_lower:
        tech_keywords.append("Excel")
    if "android" in prompt_lower:
        tech_keywords.append("Android")
    if "python" in prompt_lower:
        tech_keywords.append("Python")
    if "javascript" in prompt_lower or "js" in prompt_lower:
        tech_keywords.append("JavaScript")
    if "api" in prompt_lower:
        tech_keywords.append("API")
    
    # Extract purpose/functionality
    purpose_keywords = []
    if "track" in prompt_lower or "tracking" in prompt_lower:
        purpose_keywords.append("tracking")
    if "alert" in prompt_lower or "notify" in prompt_lower or "notification" in prompt_lower:
        purpose_keywords.append("notifications")
    if "expense" in prompt_lower or "budget" in prompt_lower:
        purpose_keywords.append("expense management")
    if "attendance" in prompt_lower:
        purpose_keywords.append("attendance management")
    if "assignment" in prompt_lower:
        purpose_keywords.append("assignment tracking")
    if "deadline" in prompt_lower:
        purpose_keywords.append("deadline management")
    
    # Generate improved suggestion based on content
    if "google sheets" in prompt_lower and "formula" in prompt_lower:
        if "expense" in prompt_lower and "alert" in prompt_lower:
            return "Create a comprehensive Google Sheets formula system that tracks monthly expenses by category, calculates running totals, and includes conditional formatting to alert you when spending exceeds predefined monthly limits for each category."
        else:
            return "Create a detailed Google Sheets formula that [specific function] with clear step-by-step instructions and examples."
    
    elif "android app" in prompt_lower or "android" in prompt_lower:
        if "student" in prompt_lower and "attendance" in prompt_lower:
            return "Design a comprehensive Android app for students that includes class attendance tracking with QR codes, assignment management with due dates, grade tracking, and push notifications for upcoming deadlines and class reminders."
        else:
            return "Design a detailed Android app concept that [specific functionality] with user interface mockups, feature specifications, and technical requirements."
    
    elif "python" in prompt_lower:
        if "api" in prompt_lower:
            return "Write a complete Python script that integrates with [specific API] to [specific functionality], including error handling, authentication, and example usage."
        else:
            return "Write a comprehensive Python script that [specific functionality] with proper documentation, error handling, and example usage."
    
    # Generic improvements for vague prompts
    elif "write something" in prompt_lower or len(prompt) < 20:
        return "Provide specific details about what you want to create, including the purpose, target audience, required features, and any technical constraints or preferences."
    
    # For prompts that are somewhat clear but could be more specific
    elif clarity_score < 4 or completeness_score < 4:
        # Try to enhance the existing prompt
        enhanced = prompt
        if not any(word in prompt_lower for word in ["specific", "detailed", "comprehensive"]):
            enhanced = enhanced.replace("Create", "Create a detailed").replace("Design", "Design a comprehensive").replace("Write", "Write a complete")
        
        if not any(word in prompt_lower for word in ["include", "with", "that"]):
            enhanced += " with specific examples and step-by-step instructions."
        
        return enhanced
    
    # Default fallback
    else:
        return prompt

def is_valid_model_response(response):
    """Check if the model response is valid and not just repeating the input"""
    if not response or len(response.strip()) < 10:
        return False
    
    # Check if response contains expected sections
    has_scores = "Scores:" in response
    has_comments = "Comments:" in response
    has_suggested = "Suggested:" in response
    
    return has_scores and has_comments and has_suggested

def get_cached_response(prompt, context):
    """Get cached response for consistent results"""
    cache_key = hashlib.md5(f"{prompt}|{context}".encode()).hexdigest()
    return response_cache.get(cache_key)

def cache_response(prompt, context, response):
    """Cache response for future use"""
    cache_key = hashlib.md5(f"{prompt}|{context}".encode()).hexdigest()
    response_cache[cache_key] = response
    
    # Limit cache size to prevent memory issues
    if len(response_cache) > 1000:
        # Remove oldest entries
        keys_to_remove = list(response_cache.keys())[:100]
        for key in keys_to_remove:
            del response_cache[key]

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

        # Check cache first for consistency
        cached_response = get_cached_response(user_prompt, context)
        if cached_response:
            return jsonify({"response": cached_response})

        # Try model generation with balanced parameters
        if model_loaded:
            try:
                # ✅ Build prompt in same style you trained
                full_prompt = f"Judge this prompt: {user_prompt}\nContext: {context if context else 'N/A'}"

                # ✅ Tokenize & generate with balanced parameters
                inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=256)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.3,  # Lower temperature for more consistency
                        top_p=0.85,       # Slightly lower for more focused sampling
                        top_k=30,         # Lower for more predictable results
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Lower penalty
                        length_penalty=1.0,
                        early_stopping=True,
                        no_repeat_ngram_size=2   # Prevent repetition of 2-grams
                    )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the result - remove the input prompt if it appears in output
                if full_prompt in result:
                    result = result.replace(full_prompt, "").strip()
                
                # Validate the response
                if is_valid_model_response(result):
                    # Cache the response for consistency
                    cache_response(user_prompt, context, result)
                    return jsonify({"response": result})
                else:
                    print(f"⚠️ Model response invalid, using fallback: {result}")
                    
            except Exception as e:
                print(f"❌ Model generation failed: {e}")
        
        # Use fallback if model fails or response is invalid
        fallback_result = create_fallback_response(user_prompt, context)
        # Cache fallback response too
        cache_response(user_prompt, context, fallback_result)
        return jsonify({"response": fallback_result})
        
    except Exception as e:
        print(f"Error in generate: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Clear the response cache to allow new variations"""
    global response_cache
    response_cache.clear()
    return jsonify({"message": "Cache cleared successfully"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860) 