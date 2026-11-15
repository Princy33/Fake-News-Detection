\# 📰 Fake News Detection \& Ethical AI Headline Generator



<p align="center">

&#x20; <strong>🤖 An AI-powered application for fake news detection and responsible headline generation.</strong>

</p>



<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />

&#x20; <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit" />

&#x20; <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" />

&#x20; <img src="https://img.shields.io/badge/GenAI-GPT--2-purple" />

&#x20; <img src="https://img.shields.io/badge/Status-Completed-success" />

</p>



\---



\## 🌟 Project Overview



The \*\*Fake News Detection \& Ethical AI Headline Generator\*\* is an AI-based web application that combines machine learning and generative AI to demonstrate two important capabilities:



\- 🔍 \*\*Fake News Detection:\*\* Classifies news content as REAL or FAKE using a Logistic Regression model and TF-IDF vectorization.

\- 🤖 \*\*AI Headline Generation:\*\* Generates news-style headlines using a GPT-2 family language model.

\- ⚖️ \*\*Responsible AI:\*\* Demonstrates ethical safeguards through content warnings, transparency, and educational disclaimers.



The project aims to demonstrate both the potential benefits and risks of Generative AI in the news domain.



\## 🎯 Project Objectives



\- Detect potentially misleading or fake news content.

\- Generate AI-based news headlines from user-provided topics.

\- Demonstrate the integration of Machine Learning and Generative AI.

\- Highlight the risks of AI-generated misinformation.

\- Promote responsible and ethical AI usage.



\## ✨ Key Features



\### 🔍 1. Fake News Detection

\- Accepts news headlines or article text.

\- Uses TF-IDF for text feature extraction.

\- Uses Logistic Regression for classification.

\- Displays REAL or FAKE prediction.

\- Shows prediction probability percentages.



\### 🤖 2. AI Headline Generator

\- Generates headlines from user-provided topics.

\- Supports two writing tones:

&#x20; - 📰 Neutral

&#x20; - ⚡ Sensational

\- Uses GPT-2 family text-generation models.

\- Displays a warning that generated content may not be factual.



\### ⚖️ 3. Ethics \& Responsible AI

\- AI-generated content warning labels.

\- Confidence score transparency.

\- Separate fake news detection system.

\- Educational-purpose disclaimer.

\- No automatic publishing of generated content.



\## 🧠 System Architecture



```text

&#x20;            👤 USER

&#x20;               |

&#x20;               v

&#x20;      🌐 STREAMLIT WEB APP

&#x20;               |

&#x20;       +-------+-------+

&#x20;       |               |

&#x20;       v               v

&#x20;🔍 NEWS DETECTION   🤖 HEADLINE GENERATION

&#x20;       |               |

&#x20;       v               v

&#x20;  TF-IDF Vectorizer   GPT-2 Family Model

&#x20;       |               |

&#x20;       v               v

&#x20;Logistic Regression   Generated Headline

&#x20;       |               |

&#x20;       v               v

&#x20; REAL / FAKE       AI Content Warning

&#x20;       |               |

&#x20;       +-------+-------+

&#x20;               |

&#x20;               v

&#x20;      ⚖️ RESPONSIBLE AI

```



\## 🛠️ Technologies Used



| Technology | Purpose |

|---|---|

| Python | Core programming |

| Streamlit | Web application interface |

| Pandas | Dataset handling |

| Scikit-learn | Machine learning |

| TF-IDF | Text feature extraction |

| Logistic Regression | Fake news classification |

| Hugging Face Transformers | Text generation |

| GPT-2 / DistilGPT-2 | Language generation |

| PyTorch | Deep learning framework |

| Joblib | Saving and loading ML models |

| Jupyter Notebook | Experimentation |



\## 📂 Project Structure



```text

Fake-News-Project/

│

├── 📄 App.py

├── 📓 app.ipynb

├── 📄 train\_classifier.py

├── 📄 train\_generator.py

├── 📄 fine\_tune\_generator.py

│

├── 📊 fake.csv

├── 📊 true.csv

│

├── 🤖 lr\_model.jb

├── 🔤 vectorizer.jb

│

├── ⚖️ ethics.\_report.md

├── 📄 requirements.txt

└── 📄 .gitignore

```



\## ⚙️ Machine Learning Workflow



1\. Load fake and real news datasets.

2\. Assign labels:

&#x20;  - `0` = FAKE

&#x20;  - `1` = REAL

3\. Handle missing values.

4\. Combine news titles and article text.

5\. Split data into training and testing sets.

6\. Apply TF-IDF vectorization.

7\. Train Logistic Regression.

8\. Evaluate the classifier using a classification report.

9\. Save the trained model and vectorizer using Joblib.



\## 🤖 Generative AI Workflow



The headline generation component uses GPT-2 family language models.



\*\*Training process:\*\*

1\. Load fake and real news datasets.

2\. Extract news headlines.

3\. Load GPT-2 tokenizer and model.

4\. Tokenize the headline dataset.

5\. Fine-tune the language model.

6\. Save the fine-tuned model and tokenizer.



The application currently loads `distilgpt2` directly in `App.py`, while the separate training script supports a fine-tuned GPT-2 model.



\## ⚖️ Ethical Considerations



Generative AI can create convincing but misleading information. This project highlights the following risks:



\- Mass misinformation.

\- Political manipulation.

\- Public panic.

\- Transformer hallucinations.

\- Dataset bias.



\*\*Safeguards:\*\*

\- AI-generated content labeling.

\- Fake news detection classifier.

\- Prediction probability transparency.

\- Educational-purpose disclaimer.

\- No automatic content publishing.



\## 🚀 Installation \& Setup



\### 1. Clone the Repository



```bash

git clone https://github.com/Princy33/Fake-News-Detection.git

```



\### 2. Navigate to the Project



```bash

cd Fake-News-Detection

```



\### 3. Create Virtual Environment



```bash

python -m venv venv

```



\### 4. Activate Environment



\*\*Windows:\*\*



```bash

venv\\Scripts\\activate

```



\### 5. Install Dependencies



```bash

pip install -r requirements.txt

```



\### 6. Run the Application



```bash

streamlit run App.py

```



Open the local URL displayed in your terminal.



\*\*Note:\*\* The application requires internet access on first use to download the DistilGPT-2 model unless it is already cached. The classifier model and vectorizer files must also be present in the project directory.



\## 📊 Model Information



| Component | Model / Method |

|---|---|

| Text Feature Extraction | TF-IDF |

| Classification Algorithm | Logistic Regression |

| Classification Labels | REAL / FAKE |

| Text Generation | DistilGPT-2 in the web app |

| Fine-tuning Script | GPT-2 |



\## ⚠️ Limitations



\- Predictions depend on the quality and coverage of training data.

\- The classifier may produce incorrect predictions.

\- Dataset bias can affect classification results.

\- Generated headlines may contain fabricated information.

\- The system cannot guarantee 100% fake news detection accuracy.

\- Model predictions should not replace independent fact-checking.



\## 🔮 Future Enhancements



\- 🌐 Real-time news verification.

\- 🧠 Advanced transformer-based classification.

\- 📈 Interactive model performance dashboard.

\- 🔗 Integration with trusted fact-checking sources.

\- 🌍 Multilingual fake news detection.

\- 🛡️ Improved explainability and responsible AI safeguards.



\## 👩‍💻 Author



\*\*Princy Mishra\*\*



\- GitHub: \[@Princy33](https://github.com/Princy33)



\---



<p align="center">

&#x20; ⭐ If you find this project useful, consider giving it a star!

</p>



<p align="center">

&#x20; <strong>⚖️ Built for AI Awareness, Misinformation Detection \& Responsible Generative AI.</strong>

</p>

