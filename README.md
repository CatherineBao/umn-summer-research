## **Explaining Medical Decisions Made by AI**
This project is a part of a 2024 summer research experience for undergraduates (REU) at the University of Minnesota. The REU's focus is human-centered computing for social good, and I am advised by Professor Qianwen Wang.
### <ins>Project description</ins>
Our research group is interested in studying how large language models (LLMs) respond differently to various medical-related inquiries. Specifically, we are studying the difference in how LLMs respond to medical inquiries formulated in a professional tone versus a demographic-specific tone. The LLMs we plan to use are GPT and Llama 3. To test this, we will feed various types of medical-related questions into these LLMs in three stages. 
#### _Stage 1_
The input is questions from the Medical Question Answering Dataset (MedQuAD). This dataset contains 47,457 questions obtained from 12 NIH websites. We will feed these questions into the LLMs we are testing and record the LLMs' responses. Since the input is professional medical-related questions, this stage of our research will serve as a baseline.
#### _Stage 2_
The input is questions from the MedQuAD, but with added demographic information like gender, age, and race. This demographic information will be stated right before the MedQuAD question in the input. We will feed these questions into the LLMs we are testing and record the LLMs' responses. 
#### _Stage 3_
This stage has two sub-stages. First, we feed a MedQuAD question into an LLM and ask the AI to generate a question that asks the same underlying question as the MedQuAD question but in the tone of a specified community. Then, we feed the AI-generated question into the LLM again and record each LLMs' response.
#### <ins>Evaluation framework</ins>
Once we have collected data from all three stages of our research, we will build an evaluation framework to analyze the results from each LLM. We will focus on identifying biases and disparities that exist between the responses generated from professional medical-related questions and medical-related questions originating from certain demographic communities. Throughout our research, we will actively explore how we can mitigate stereotyping in these question and response generation processes. 