# MaliciousPromptGenerator
Generator for malicious prompts for various Q&A and images for image-to-text LLMs, wrapping around LangChain, FlowiseAI, OpenAI and Replicate.\
\
**Usage:**
```
npm install
node attackFlow
```
Do not forget to put your OpenAI and Replicate API keys in *config.json*. Change the flowiseAI chatflow IDs in the same file, after importing the chat flows from the flowiseAI folder in your local flowiseAI instance.\
\
Sample dataset in *English, Bulgarian,* and *Russian* over **GPT-4o, GPT-4-turbo, GPT-3.5.-turbo, llava-v1.6-mistral-7b, llava-v1.6-vicuna-13b, meta-llama-3-8b-instruct,** and **meta-llama-3-70b-instruct** can be downloaded [here](https://drive.google.com/file/d/1gAYx7yExJhLr61LvFWxbCQga2wgJ7pwc/view?usp=sharing).
