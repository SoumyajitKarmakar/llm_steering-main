import os
import random
from utils import LLMType

# =============================================================================
# CONCEPT CATEGORIES AND THEIR PROMPT TEMPLATES
# =============================================================================

def pca_adjective_dataset(llm, mood, category, seed=0, same_data=False):
    tokenizer = llm.tokenizer
    concept_type = mood

    data_dir = 'data/general_statements_adj/{}/'.format(category)
    # data_dir = 'data/general_statements' # old

    random.seed(0)

    print(f"Using {data_dir}")

    if category == "complexity":
        user_str = 'Describe the following task, concept, or system, emphasizing it being {concept_type}. \nTopic: {statement}'
        default_str = 'Describe the following task, concept, or system. \nTopic: {statement}'
    elif category == "logic":
        user_str = 'Analyze the following assertion, treating it as {concept_type}. \nAssertion: {statement}'
        default_str = 'Analyze the following assertion. \nAssertion: {statement}'
    elif category == "physical":
        user_str = 'Write a description of the following item that emphasizes it being {concept_type}. \nItem: {statement}'
        default_str = 'Write a description of the following item. \nItem: {statement}'
    elif category == "social":
        user_str = 'Adopt a {concept_type} persona. What are your thoughts on the following statement? \nStatement: {statement}'
        default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'

        # old
        # user_str = 'Take on a {concept_type} mood.  What are your thoughts on the following statement? \nStatement: {statement}'
        # default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    elif category == "state":
        user_str = 'Describe the current state of the following item, focusing on it being {concept_type}. \nItem: {statement}'
        default_str = 'Describe the current state of the following item. \nItem: {statement}'
    elif category == "texture":
        user_str = 'Describe the sensory experience of interacting with the following item, emphasizing it being {concept_type}. \nItem: {statement}'
        default_str = 'Describe the sensory experience of interacting with the following item. \nItem: {statement}'
    elif category == "time":
        user_str = 'Describe the following event or entity, emphasizing it being {concept_type}. \nSubject: {statement}'
        default_str = 'Describe the following event or entity. \nSubject: {statement}'
    

    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
        raw_data = f.readlines()
    
    if same_data:
        print("Using same data for both classes.")
        raw_data_2 = raw_data
    else:
        with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    csp_data = [user_str.format(concept_type=mood, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    for idx, s in enumerate(csp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()





    print(f"Concept: {concept_type}")
    print(f"Category: {category}")
    print(f"Reading from: {data_dir}")
    print(f"Sample positive prompt: {csp_data[0][:200]}...")
    print(f"Sample negative prompt: {ncsp_data[0][:200]}...")



    return formatted_data


if __name__ == "__main__":

    print("need to load llm lol!")



