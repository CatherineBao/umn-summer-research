import openai
import os
from dotenv import load_dotenv
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

def askGipity(system, user):
    model = "gpt-4"
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response = openai.chat.completions.create(
        model=model,
        messages=prompt,
        temperature = 0.3 
    )
    return response.choices[0].message.content

def read_data():
    all_data = {"train": "data/train-00000-of-00001-a6e6f83961340098.parquet", 
                "test": "data/test-00000-of-00001-997553f5b24c9767.parquet"}
    df = pd.read_parquet("hf://datasets/augtoma/medqa_usmle/" + all_data["train"])
    return df

def extract_column(df, column):
    return df[column].tolist()

def is_diagnosis_related(a):
    isDiagnosis = askGipity("If the provided item is a specific unique type of disease return true with no other commentary else return false in all lowercase. Return false if it is a symptom of diseases rather than the disease itself. Return false if the question references an image.", a)
    return isDiagnosis == "true"

def get_random(number, question, answer):
    random.seed(10)
    valid_pairs = []
    indices = list(range(len(question)))

    while len(valid_pairs) < number:
        sampled_index = random.sample(indices, 1)[0]
        q, a = question[sampled_index], answer[sampled_index]
        if is_diagnosis_related(a):
            valid_pairs.append((q, a))
        indices.remove(sampled_index)

    return valid_pairs

def run_accuracy_test(answers, responses):
    results = []
    for answer, response in zip(answers, responses):
        system_prompt = f"""Given {answer} is the correct diagnosis for the user. Does the response mention 
            the correct or simular diagnosis? Respond accurate or inaccurate.
            Compared to the answer, determine if the response is accurate or 
            inaccurate. An example of a correct but not exact answer would be if Invasive lobular carcinoma is 
            the correct diagnosis and the response mentions breast cancer.
            Respond only with accurate or inaccurate. Do not explain your 
            reasoning. The response is accurate if the correct diagnosis is provided anywhere
            in the answer."""
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

    surveyAnswers = [askGipity(
        f"""Answer the Additional Information Section with common symptoms of {answer}. Answer in a casual tone used by the general public, do not include details that a patient would only know through medical tests or a doctor's visit.
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

def systemDiagnosis(question):
    systemMessage = """If enough specific information is provided from the user, please provide the following information:\n
        1. Provide the most likely diagnosis relating to the user's symptoms. (IMPORTANT)\n
        2. Provide other alternative diagnosis possibilities \n
        3. Provide steps and options to treat the symptoms moving forward including first-aid/home treatment plans and recommending a specialist (include what kind of doctor they should visit)\n
        4. Provide links and websites to other sources where more information about the most likely diagnosis can be found\n
        5. Ask for any follow-up questions involving the user's condition (IMPORTANT)\n
        6. Provide more information as needed such as possible medicine options and treatment and safety considerations or other points relevant to the user's question with links to specific pages\n
        \n
        Formatting Instruction:\n
        Diagnosis - Diagnosis Title \n
        Information\n
        Title:\n
        Information\n
        Title:\n
        Information\n
        ...\n
        \n
        Example (Refer to the example provided below for formatting and content): \n
        Q: \n
        28M, 168lbs, 6'0. Since last year I've noticed that my body doesn't seem to be healing from injuries that are more than skin deep. I developed a herniated disc two months ago, in November I seemed to have sprained my foot and hand or at least I'm guessing they're sprains based on how I got those injuries((it's been 7 months and they have barely improved). Two weeks ago I developed some kind of swelling behind my knee. I read online that the swelling could be a Baker's Cyst, caused by a knee injury or arthritis(I think it might be an injury from standing on my knee once). I haven't exercised since March to try and give my disc time to recover(not much improvement) so it isn't exercise that could have caused the cyst. Any ideas of what could be the underlying problem?
        Additional Information:\n
        Age: No\n
        Medical History: I have diabetes and I had my appendix out when I was 9\n
        Occupation: I'm a teacher, and I constantly have to bend over and pick up the chalk that children throw at me.\n
        Family History: Not that I know of\n
        Pain and Function: I am not able to bend over as much\n
        A: \n
        Diagnosis - Symptoms caused by Diabetes:\n
        Considering your history of injuries that are not healing properly, along with the development of a herniated disc, foot and hand sprains, and swelling behind your knee, there may be an underlying issue affecting your body's healing capability. One possible explanation for your prolonged recovery and multiple injuries could be related to a systemic condition such as diabetes. Diabetes can impact the body's ability to heal wounds and injuries efficiently, leading to delayed healing and increased susceptibility to injuries.\n
        \n
        Other Possible Considerations:\n
        Nutritional Deficiencies: Lack of certain vitamins and minerals, such as Vitamin D, calcium, magnesium, and Vitamin C, can impair the body’s ability to heal properly.\n
        Chronic Inflammation: Conditions like chronic inflammation can slow down the healing process. This can be due to autoimmune disorders, chronic infections, or even lifestyle factors such as diet and stress.\n
        Circulatory Issues: Poor circulation can affect healing, as it reduces the supply of necessary nutrients and oxygen to injured areas. Conditions like diabetes or vascular diseases can contribute to this.\n
        Hormonal Imbalances: Hormones play a crucial role in tissue repair. Imbalances in thyroid hormones, cortisol, or testosterone can impair healing.\n
        Infection: Sometimes, an infection in the injured area can cause persistent pain and swelling, preventing proper healing.\n
        \n
        Home-Treatments:\n
        1. Manage blood sugar levels: Ensure you are actively monitoring and managing your blood sugar levels through proper diet, regular exercise (if approved by your healthcare provider), and any prescribed medications.\n
        2. Support wound healing: Focus on maintaining good wound care practices for any open injuries or wounds to prevent infections and facilitate healing.\n
        3. Modify activities: Consider adjusting your teaching duties to minimize bending over and lifting heavy objects to reduce strain on your body and allow for better recovery.\n
        \n
        Moving Forward:\n
        Given your history of diabetes and the issues with slow healing and multiple injuries, it is crucial to seek medical evaluation to address these concerns. Proper management of diabetes and appropriate treatment for your current injuries are essential to prevent further complications.\n
        Warning signs to watch out for:\n
        Sudden Energy Crash: Be alert to a sudden drop in energy, which could indicate an underlying issue that needs immediate attention.\n
        Mental Health Changes: Any changes in mood, such as irritability, anxiety, or depression, should be addressed promptly.\n
        Physical Symptoms: New symptoms like palpitations, dizziness, or significant changes in vision or cognition should be evaluated by a healthcare provider immediately.\n
        \n
        Recommendation for Specialist:\n
        Consulting with a healthcare provider, preferably a primary care physician or an orthopedic specialist, would be beneficial for a comprehensive evaluation of your musculoskeletal issues and consideration of your diabetes. Additionally, you may benefit from a referral to a podiatrist for the foot injury and a rheumatologist to assess the swelling behind your knee in case it is related to arthritis.\n
        \n
        Additional Notes:\n
        If you require assistance in finding affordable healthcare options, consider reaching out to local clinics, community health centers, or healthcare assistance programs such as...\n
        \n
        For more information on diabetes and wound healing, you can visit the American Diabetes Association website: https:/ \n
        Do you have any additional questions about your condition?
        """
    return askGipity(systemMessage, question)

def main():
    df = read_data()
    diagnosisSet = list(get_random(20, extract_column(df, "question"), extract_column(df, "answer")))

    questions = [item[0] for item in diagnosisSet]
    answers = [item[1] for item in diagnosisSet]

    casualPhrasing = [askGipity("Rephrase this question so that it is similar to that of the general public (similar to Reddit posts) in a first-person point of view but exclude any information that the general public wouldn't have without going to a doctor for tests.", q) for q in questions]
    print("All questions have successfully converted to casual phrasing!")
    questionWithSurveyResult = questionWithSurvey(casualPhrasing, answers)
    print("All questions have successfully been processed with additional information!")

    with ThreadPoolExecutor() as executor:
        defaultAnswer = list(executor.map(
                        lambda q: askGipity("Address the inquiry provided by the user", q), questions
                        ))
        scoreDefult = run_accuracy_test(answers, defaultAnswer)
        print("Default Score: " + str(getScore(scoreDefult)))

    with ThreadPoolExecutor() as executor:
        originalAnswer = list(executor.map(
                        lambda q: askGipity("Address the inquiry provided by the user", q), casualPhrasing
                        ))
        scoreOG = run_accuracy_test(answers, originalAnswer)
        print("Original Score With Casual Tone: " + str(getScore(scoreOG)))
    

    with ThreadPoolExecutor() as executor:
        ogWithSystem = list(executor.map(
                        lambda q: systemDiagnosis(q), casualPhrasing
                        ))
        scoreOGWithSystem = run_accuracy_test(answers, ogWithSystem)
        print("System Score With Casual Tone: " + str(getScore(scoreOGWithSystem)))

    with ThreadPoolExecutor() as executor:
        answerWithSurvey = list(executor.map(
                        lambda q: askGipity("Address the inquiry provided by the user", q), questionWithSurveyResult
                        ))
        scoreSurvey = run_accuracy_test(answers, answerWithSurvey)
        print("Original Score With Survey: " + str(getScore(scoreSurvey)))

    with ThreadPoolExecutor() as executor:
        surveyAndSystem = list(executor.map(
                        lambda q: systemDiagnosis(q), questionWithSurveyResult
                        ))
        scoreSurveyAndSystem = run_accuracy_test(answers, surveyAndSystem)
        print("System Score With Survey: " + str(getScore(scoreSurveyAndSystem)))
    
    data = {
        "question": questions,
        "answer": answers,
        "generalPublicQuestion": casualPhrasing,
        "surveyQuestions": questionWithSurveyResult,
        "defaultAnswer": defaultAnswer,
        "defaultScore": scoreDefult,
        "generalPublicAnswerOG": originalAnswer,
        "OGscore": scoreOG,
        "ogWithSystemAnswer": ogWithSystem,
        "scoreOGWithSystem": scoreOGWithSystem,
        "surveyAnswer": answerWithSurvey,
        "surveyScore": scoreSurvey,
        "surveyAndSystemAnswer": surveyAndSystem,
        "surveyAndSystemScore": scoreSurveyAndSystem,
    }

    df_output = pd.DataFrame(data)
    df_output.to_csv("testResults.csv", index=False)

if __name__ == "__main__":
    main()