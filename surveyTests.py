import openai
import os
from dotenv import load_dotenv
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

def askGipity(system, user):
    model = "gpt-4"
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response = openai.chat.completions.create(
        model=model,
        messages=prompt
    )
    return response.choices[0].message.content

def read_data():
    all_data = {"train": "data/train-00000-of-00001-a6e6f83961340098.parquet", 
                "test": "data/test-00000-of-00001-997553f5b24c9767.parquet"}
    df = pd.read_parquet("hf://datasets/augtoma/medqa_usmle/" + all_data["train"])
    return df

def extract_column(df, column):
    return df[column].tolist()

def get_random(number, data):
    random.seed(100)
    return random.choices(data, k=number)

def run_accuracy_test(answers, responses):
    system_prompt = """You will be given a ground truth answer and a response to a medical question. 
                        Compared to the ground truth answer, determine if the response is accurate or 
                        inaccurate. Respond only with accurate or inaccurate. Do not explain your 
                        reasoning. The response is accurate if the correct diagnosis is provided
                        in the answer."""
    results = []
    for answer, response in zip(answers, responses):
        user_prompt = f"Ground truth answer: {answer}. Response: {response}"
        result = askGipity(system_prompt, user_prompt)
        results.append(result)
    return results

def getScore(accuracyResults):
    totalScore = sum(1 for result in accuracyResults if result == "Accurate")
    return totalScore / len(accuracyResults)

def questionWithSurvey(casualPhrasing, answers):
    surveyQuestions = [askGipity(
        """ Create a list of additional information, questions, or symptoms that a doctor will ask to create a more accurate diagnosis. These should cover age, gender, race, other symptoms, and relevant details.

            When the patient mentions visiting or consulting a doctor, you must ask for any exam results (vitals, blood pressure, temperature, pulse, respiration, etc.).

            Here are the areas to cover and DO NOT ask about information that the user has already answered:
            Previous Health Issues: Ask about any past health conditions.
            Lifestyle: Inquire about smoking, alcohol use, and exercise habits.
            Doctor Visits: If the patient mentions a doctor at all, ask for any physical examination or lab results.
            Specific Symptoms: Tailor questions to the patient's mentioned symptoms. For example, if a male patient reports urinary problems, ask if it hurts when he pees.
            Ensure to follow each topic with a brief description if needed. Aim for short and concise questions. DO NOT provide any additionary commentary.
            Example: 
            Q: I got a cut a few weeks ago and it hasn’t healed yet recently it started hurting more than usual and it looks red on the outside. Should I be concerned?
            A: 
            Additional Information:
            Is the redness spreading or forming a red streak? Can you describe the appearance of the redness?:
            Is there swelling, warmth, pain, or tenderness in the area of the cut?:
            Is there any pus forming around or oozing from the wound?:
            Do you have swollen lymph nodes in the neck, armpit, or groin?:
            Do you have a fever or other new developments to note?:

            Notes:
            Don’t include any additional commentary or formatting outside of specifications.
            Don’t repeat points already stated in the original question.
            Ask about basic personal information such as age, gender, weight, and race if relevant.
            Don't ask for information the user has already provided. 
        """, 
    question) for question in casualPhrasing]

    surveyAnswers = [askGipity(f"""
        Answer the Additional Information Section with common symptoms of {answer}. Answer in a casual tone used by the general public, do not include details that a patient would only know through medical tests or a doctor's visit.
        Formatting: Include the Original Prompt in the answer returned
        Example: 
        Q: Additional Information:
        Is the redness spreading or forming a red streak? Can you describe the appearance of the redness?: 
        Is there swelling, warmth, pain, or tenderness in the area of the cut?: 
        Is there any pus forming around or oozing from the wound?: 
        Do you have swollen lymph nodes in the neck, armpit, or groin?: 
        Do you have a fever or other new developments to note?: 

        A: Additional Information:
        Is the redness spreading or forming a red streak? Can you describe the appearance of the redness?: There is a red streak forming on the cut area and it is spreading out in a rash like apparence.
        Is there swelling and tenderness in the area of the cut?: There is some swelling and tenderness
        Is there any pus forming around or oozing from the wound?: No
        Do you have swollen lymph nodes in the neck, armpit, or groin?: No
        Do you have a fever or other new developments to note?: No
        """, question) for question, answer in zip(surveyQuestions, answers)]
    
    return [question + survey for question, survey in zip(casualPhrasing, surveyAnswers)]

def main():
    df = read_data()
    questions = get_random(20, extract_column(df, "question"))
    answers = get_random(20, extract_column(df, "answer"))

    with ThreadPoolExecutor() as executor:
        casualPhrasing = list(executor.map(
            lambda q: askGipity("Rephrase this question so that it is simular to that of the general public (simular to reddit posts) in a first person point of view but exclude any information that the general public wouldn't have without going to a doctor for tests.", q), 
            questions
        ))

        originalAnswer = list(executor.map(
            lambda q: askGipity("Address the inquiry provided by the user", q),
            casualPhrasing
        ))

    questionWithSurveyResult = questionWithSurvey(casualPhrasing, answers)
    
    with ThreadPoolExecutor() as executor:
        answerWithSurvey = list(executor.map(
            lambda q: askGipity("Address the inquiry provided by the user", q), 
            questionWithSurveyResult
        ))

    scoreOG = run_accuracy_test(answers, originalAnswer)
    scoreSurvey = run_accuracy_test(answers, answerWithSurvey)
    print(getScore(scoreOG), getScore(scoreSurvey))

    data = {
        "question": questions,
        "answer": answers,
        "generalPublicQuestion": casualPhrasing,
        "generalPublicAnswerOG": originalAnswer,
        "OGscore": scoreOG,
        "surveyAnswer": answerWithSurvey,
        "surveyScore": scoreSurvey
    }

    df_output = pd.DataFrame(data)
    df_output.to_csv("testResults.csv", index=False)

if __name__ == "__main__":
    main()