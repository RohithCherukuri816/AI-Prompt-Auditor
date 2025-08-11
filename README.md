# AI Prompt Auditor

A Flask-based web application that uses a fine-tuned Flan-T5 model to evaluate and judge AI prompts based on clarity, completeness, and alignment criteria.

## 🚀 **Quick Start**

```bash
# Activate virtual environment
myenv\Scripts\activate

# Run the application
python app.py
```

Then open your browser to `http://localhost:7860`

## 📁 **Project Structure**

```
AI Prompt Auditor/
├── app.py                    # ✅ Main Flask application
├── requirements.txt          # ✅ Python dependencies
├── README.md                # ✅ Project documentation
├── prompt_judge_dataset.jsonl  # ✅ Training dataset
├── templates/
│   └── index.html           # ✅ Modern UI template
├── finetuned-flan-t5/       # ✅ Fine-tuned model
└── myenv/                   # ✅ Virtual environment
```

## ✨ **Features**

### **Modern UI**
- **Dark/Light mode toggle** with persistent settings
- **Responsive design** that works on all devices
- **Beautiful animations** and smooth transitions
- **Accessible** with keyboard navigation and screen reader support

### **Smart Prompt Evaluation**
- **Multi-criteria scoring** (clarity, completeness, alignment)
- **Context-aware analysis** with industry and audience considerations
- **Intelligent suggestions** for prompt improvement
- **Response caching** for consistency

### **Fallback System**
When the model fails, the system provides intelligent rule-based evaluation:
- **Length-based scoring** (short prompts get low scores)
- **Content-based analysis** (specific keywords improve scores)
- **Context-aware suggestions** (professional/friendly tone detection)

## 🔧 **How It Works**

### **1. Prompt Evaluation**
The system evaluates prompts based on:
- **Clarity**: How specific and clear the prompt is
- **Completeness**: Whether it includes necessary details
- **Alignment**: How well it matches the intended audience/context

### **2. Caching System**
- **Response Caching**: Same prompt + context = same response
- **Cache Control**: "Clear Cache" button for fresh variations
- **Memory Management**: Automatic cache cleanup

### **3. Fallback System**
Provides intelligent prompt evaluation when the model fails:
- **Length-based**: Short prompts (<20 chars) get low scores
- **Content-based**: "Write something" gets low scores, "clear detailed" gets high scores
- **Context-aware**: Professional/friendly context improves alignment scores

## 🎯 **Example Prompts to Try**

### **Good Prompts** (should get high scores)
- "Write a detailed Python script that fetches weather data from the OpenWeatherMap API and displays the temperature for a given city."
- "Design a mobile app interface for a fitness tracker that logs daily steps, calories burned, and sends motivational notifications."

### **Bad Prompts** (should get low scores)
- "Write something."
- "Explain it."
- "Describe process."

## 🛠 **Technical Details**

### **Model Architecture**
- **Base Model**: Google Flan-T5-base
- **Fine-tuning**: Custom dataset with 501 training examples
- **Task**: Sequence-to-sequence prompt evaluation

### **Generation Parameters**
- **Temperature**: 0.3 (balanced consistency)
- **Top-p**: 0.85 (focused sampling)
- **Top-k**: 30 (predictable results)
- **Repetition Penalty**: 1.1 (prevents loops)

## 🔍 **Troubleshooting**

### **If Model Generates Same Output**
1. **Use the Clear Cache button** to get fresh variations
2. **Check the console logs** for error messages
3. **Ensure the model is properly loaded** on startup

### **If Fallback System Fails**
1. **Check logs** for error messages
2. **Validate input** - ensure prompt is not empty
3. **Restart the application** if needed

## 📊 **Performance Metrics**

### **Current Features**
- ✅ Fallback system provides consistent evaluation
- ✅ Caching system maintains consistency
- ✅ Modern UI with dark/light mode
- ✅ Responsive design for all devices

### **Expected Behavior**
- **Good prompts** get scores 4/4/4 or 4/4/3
- **Bad prompts** get scores 1/1/2 or 2/2/3
- **Consistent responses** for same inputs
- **Varied responses** when cache is cleared

## 🤝 **Contributing**

To improve the model:
1. **Enhance dataset**: Add more diverse training examples
2. **Tune parameters**: Experiment with different training settings
3. **Add validation**: Create test set for evaluation
4. **Improve fallback**: Enhance rule-based scoring logic

## 📝 **License**

This project is for educational and research purposes. 