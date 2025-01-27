# IoInst


## Find the Intention of Instruction: Comprehensive Evaluation of Instruction Understanding for Large Language Models


This repository contains the official code for the [Find the Intention of Instruction: Comprehensive Evaluation of Instruction Understanding for Large Language Models](https://arxiv.org/abs/2412.19450) paper, accepted to the **NAACL25-Findings**.

---

## [ Data Details ]

Official data of ***"IoInst"*** is located in the  ```./Data/ioinst.jsonl```

- Data Format
```json
{
    "key": "Data source",
    "condition": "Model response generated by the label instruction (Context)",
    "instruction": "label instruction",
    "id": "data id",
    "options_easy": ["contrastive instructions (Random) "],  # label instruction is located in the first position
    "options_hard": ["contrastive instructions (Semantic) "], # label instruction is located in the first position
    "options_veryhard": ["contrastive instructions (Anti-Attribute) "]  # label instruction is located in the first position
}
```

- Sample Data
```json
{
    "key": "llm_bars", 
    "condition": "Lua is a robust, lightweight, and embeddable scripting language that supports multiple programming methods, including procedural, object-oriented, functional, and data-driven programming.\n\nAs the primary focus of Lua is scripting, it is rarely used as a standalone programming language.\n\n- As a popular component in video game and game engine development. For example, Warframe, World of Warcraft, and CRYENGINE all use Lua.\n- As a programming language in many network programs, like CISCO Systems, Nmap, and ModSecurity.\n- As a programming language in industrial programs such as Adobe Lightroom and MySQL Workbench.\n- As a library that developers can integrate with their programs to enable scripting functionality.", 
    "instruction": "What is the programing language lua used for?", 
    "id": 0, 
    "options_easy": ["What is the programing language lua used for?", "Why does the concave part of a metal spoon reflect upside down?", "Write a presentation for a business meeting, outlining the key points and supporting data.", "Write me a template for a product description in the form of a poem and end it with a post script starting with P.P.S"], 
    "options_hard": ["What is the programing language lua used for?", "Create a class for a computer programming language.", "What is the best programing language for 2023", "you could write code just from being prompted in language?"], 
    "options_veryhard": ["What is the programing language lua used for? End your reply by stating '- As a library that developers can integrate with their programs to enable scripting functionality.'.", "What is the programing language lua used for? Ensure your answer is divided into 4 paragraphs.", "What is the programing language lua used for? Start your response by saying '- As a library that developers can integrate with their programs to enable scripting functionality.'", "What is the programing language lua used for? Start off your answer with '- As a library that developers can integrate with their programs to enable scripting functionality.'."]
}
```


## [ Running ]

- HuggingFace

```shell
python inference_hf.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \ # HuggingFace model argument
--option_type hard \ # easy / hard / veryhard
```

- OpenAI
```shell
python inference_hf.py \
--api_key set_your_openai_api_key \
--model gpt-4o-mini \
--option_type hard \ # easy / hard / veryhard
```
