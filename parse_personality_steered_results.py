import re
import csv
from openai import OpenAI
import pickle

client = OpenAI(
    api_key=""
)

def parse_personality_responses(response):
    # print(response)
    passage = re.split(r"\|>assistant<\|end_header_id\|>", response[1])[1]
    passage = "".join(passage)

    # passage = re.split(r"<\|eot_id\|>", response[1])[2]
    # if len(passage) == 0:
    #     return ""
    # passage = re.split(r"\|>assistant<\|end_header_id\|>", passage)
    # if len(passage) == 0:
    #     return ""
    # if passage[0][0] == '<':
    #     if len(passage) <= 1: 
    #         return ""
    #     passage = passage[1]
    # else:
    #     passage = passage[0]
    # if len(passage) == 0:
    #     return ""        
    # passage = re.split("<", passage)[0]
    # passage = "".join(passage)
    return passage

def load_prompt(label, version):
    dir = 'evaluation_prompts/'
    if version == 1:
        version_label = ''
    else:
        version_label = f'_v{version}'
    if label == 'fears':
        with open(dir + f'phobia_eval{version_label}.txt', "r") as f:
            return f.read()
    elif label == 'personalities':
        with open(dir + f'personality_eval{version_label}.txt', "r") as f:
            return f.read()
    elif label == 'moods':
        with open(dir + f'mood_eval{version_label}.txt', "r") as f:
            return f.read()
    elif label == 'places':
        with open(dir + f'topophile_eval{version_label}.txt', "r") as f:
            return f.read()
    elif label == 'personas':
        with open(dir + f'persona_eval{version_label}.txt', "r") as f:
            return f.read()

def main():

    METHOD = 'prompting'
    CONCEPT_CLASSES = ['fears', 'personalities', 'moods', 'places', 'personas']
    # CONCEPT_CLASSES = ['personalities', 'moods', 'places', 'personas']
    # CONCEPT_CLASSES = ['places', 'personas']
    # CONCEPT_CLASSES = ['personas']
    VERSIONS = [1, 2, 3, 4, 5]
    # VERSIONS = [1]
    MODEL_NAME = 'llama'
    MODEL_VERSION = '3.1'
    MODEL_SIZE = '8B'

    for VERSION in VERSIONS:
        if VERSION == 1:
            VERSION_LABEL = ''
        else: 
            VERSION_LABEL = f'_v{VERSION}'

        for CONCEPT_CLASS in CONCEPT_CLASSES:
            # if CONCEPT_CLASS != 'fears':
            #     continue
            file_path = f'cached_outputs/{METHOD}_{CONCEPT_CLASS}_steered_500_concepts_llama_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.pkl'

            results = pickle.load(open(file_path, 'rb'))

            outputs = []
            for results_idx, (personality, responses) in enumerate(results):
                # if personality != 'Cheyenne':
                #     continue
                print(f"================================PERSONALITY:{personality}=============================")
                best_score = 0
                for response in responses:
                    parsed_response = ""
                    parsed_response = parse_personality_responses(response)
                    if parsed_response == "":
                        parsed_response = "None"

                    prompt_template = load_prompt(CONCEPT_CLASS, VERSION)

                    prompt = prompt_template.format(personality=personality, parsed_response=parsed_response)            
                    # print(prompt)
                    # print("------------------")
                    # print(parsed_response)
                    # return
                    output = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                },
                            ],
                            temperature=0.,
                            max_tokens=20,
                            # model="gpt-4o",
                            # model='gpt-4o-mini-2024-07-18'
                            model='gpt-4o-2024-11-20'
                        )
                    content = output.choices[0].message.content
                    score = 0
                    # print(parsed_response)
                    # print("===================")
                    # print(content)
                    # print(prompt)
                    # print(response, parsed_response, content)
                    # if "Score: " in content: 
                    #     score = int(content.split("Score: ")[1][0])
                    if "Score: " in content:
                        try:                        
                            score = int(content.split("Score: ")[1][0])
                        except (IndexError, ValueError) as e:
                            score = 0
                    best_score = max(score, best_score)
                print(personality, "Best score: ", best_score)
                outputs.append((personality, best_score))

                # break
            # return 
            output_csv = f"csvs/{METHOD}_{CONCEPT_CLASS}_gpt4o_outputs_500_concepts_{MODEL_NAME}_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.csv"
            # output_csv = f"csvs/{METHOD}_{CONCEPT_CLASS}_gpt4o_outputs_500_concepts_llama_{MODEL_VERSION}_english_only{VERSION_LABEL}.csv"

            with open(output_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["GPT-4o responses"])  # Header row
                writer.writerows(outputs)  # Write extracted data

if __name__ == "__main__":
    main()