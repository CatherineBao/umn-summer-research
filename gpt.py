import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


"""
TODO this week (and next Monday):
- Implement parallel computing: DONE
- Edit system prompt to return only the best answer to the question: DONE
- Give GPT three classes to sort response accuracy into (accurate, relevant, not accurate): DONE
- Generate classifier accuracy statistics:
- Generate response accuracy statistics for each factor introduced: 
- Generate tone specific (general public, etc.) prompts and record accuracy:

TODO when have time:
- Edit system prompt again to return only the top three best answers to the question: 
- See if I can reorganize code to make functions calls run faster: 
- Add function comments:
"""
def read_data():
    all_data = {"train": "data/train-00000-of-00001-a6e6f83961340098.parquet", 
                "test": "data/test-00000-of-00001-997553f5b24c9767.parquet"}
    df = pd.read_parquet("hf://datasets/augtoma/medqa_usmle/" + all_data["train"])
    return df

def read_questions():
    df = read_data()
    questions = []
    for index, row in df.iterrows():
        question = row["question"]
        questions.append(question)
    return questions

def read_answers():
    df = read_data()
    answers = []
    for index, row in df.iterrows():
        answer = row["answer"]
        answers.append(answer)
    return answers

def get_random(number, data):
    random.seed(100)
    result = random.choices(data, k = number)
    return result

def get_df(data, headers):
    df = pd.DataFrame(data)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def add_column_to_df(df, data, header):
    df[header] = data

def get_results_df(questions, responses, answers, accuracy): 
    df = get_df(questions, ["Question"])
    add_column_to_df(df, responses, "Response")
    add_column_to_df(df, answers, "Answer")
    add_column_to_df(df, accuracy, "Accuracy")
    return df

def make_csv(df, filename):
    df.to_csv(filename, index = False)

def query_GPT_without_system_role(user_prompt):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model = "gpt-4o"
    prompt = [{"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(model = model, messages = prompt)
    return response.choices[0].message.content

def query_GPT_with_system_role(system_prompt, user_prompt):
    client = OpenAI()
    model = "gpt-4o"
    prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(model = model, messages = prompt)
    return response.choices[0].message.content

def get_responses_without_system_role(questions):
    results = []
    for question in questions:
        response = query_GPT_without_system_role(question)
        results.append(response)
    return results 

def get_responses_with_system_role(questions, system_prompt):
    results = []
    for question in questions:
        response = query_GPT_with_system_role(system_prompt, question)
        results.append(response)
    return results

def run_accuracy_test(answers, responses):
    results = []
    index = 0
    system_prompt = """You will be given a ground truth answer and a response to a medical question. 
                        Compared to the ground truth answer, determine if the response is accurate, relevant 
                        but not fully accurate, or inaccurate. Respond only with accurate, relevant, or inaccurate. 
                        Do not explain your reasoning."""
    for response in responses:
        user_prompt = "Ground truth answer: " + answers[index] + ". Response: " + response
        result = query_GPT_with_system_role(system_prompt, user_prompt)
        results.append(result)
        index = index + 1
    return results

def get_accuracy_statistics(df):
    total_responses = 0
    accurate_responses = 0
    relevant_responses = 0
    inaccurate_responses = 0
    for index, row in df.iterrows():
        total_responses = total_responses + 1
        accuracy = row["Accuracy"]
        if accuracy == "Accurate" or accuracy == "accurate":
            accurate_responses = accurate_responses + 1
        elif accuracy == "Relevant" or accuracy == "relevant":
            relevant_responses = relevant_responses + 1
        elif accuracy == "Inaccurate" or accuracy == "inaccurate":
            inaccurate_responses = inaccurate_responses + 1
    accurate = round(accurate_responses / total_responses, 2)
    relevant = round(relevant_responses / total_responses, 2)
    inaccurate = round(inaccurate_responses / total_responses, 2)
    results = ["Accurate:", accurate, accurate_responses, "Relevant:", relevant, relevant_responses, "Inaccurate", inaccurate, inaccurate_responses]
    return results

def present_results(questions, filename):
    system_prompt = """You will be given a medical question. Provide the best answer to the question. 
                    Do not explain your reasoning, and keep your response brief."""
    responses = get_responses_with_system_role(questions, system_prompt)
    answers = get_random(20, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, filename)
    accuracy_statistics = get_accuracy_statistics(df)
    print(accuracy_statistics)

def get_baseline_questions():
    original_questions = get_random(20, read_questions())
    system_prompt = "Rewrite the question provided with age and gender removed. Change pronouns to be gender neutral."
    questions = get_responses_with_system_role(original_questions, system_prompt)
    return questions

def get_age_questions():
    original_questions = get_random(20, read_questions())
    system_prompt = "Rewrite the question provided with gender removed. Change pronouns to be gender neutral."
    questions = get_responses_with_system_role(original_questions, system_prompt)
    return questions

def get_gender_questions():
    original_questions = get_random(20, read_questions())
    system_prompt = "Rewrite the question provided with age removed."
    questions = get_responses_with_system_role(original_questions, system_prompt)
    return questions

def get_race_questions():
    questions = []
    original_questions = get_random(20, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided with age and gender removed. Change pronouns to be gender netural.
                        Add that the person's race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def get_age_gender_questions():
    return get_random(20, read_questions())

def get_age_race_questions():
    questions = []
    original_questions = get_random(20, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided with gender removed. Change pronouns to be gender netural.
                        Add that the person's race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def get_gender_race_questions():
    questions = []
    original_questions = get_random(20, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided with age removed. Add that the person's 
                        race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def get_age_gender_race_questions():
    questions = []
    original_questions = get_random(20, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided to add that the person's race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def main():
    present_results(get_race_questions(), "race.csv")
    #results = []
    #with ThreadPoolExecutor(max_workers = 8) as executor: # TOOK 3 MINS TO RUN
        #executor.submit(present_results, get_baseline_questions(), "BASELINE.csv")
        #results.append(executor.submit(present_results, get_age_questions(), "age.csv"))
    #print(results)

    """baseline = Process(target = present_results(get_baseline_questions(), "baseline.csv"))
    baseline.start()
    age = Process(target = present_results(get_age_questions(), "age.csv"))
    age.start()
    baseline.join()
    age.join()"""
    #print(present_results(get_baseline_questions(), "baseline.csv"))

if __name__ == "__main__":
    main()

