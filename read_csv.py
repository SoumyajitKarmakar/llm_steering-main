import csv 

def readfile(fname):
    total = 0
    steered = 0
    with open(fname) as f:
        for idx, line in enumerate(f): 
            if idx == 0:
                continue
            t = int(line.strip().split(',')[-1])
            steered += t
            total += 1
    print("Steered: ", steered, "Out of: ", total)
    return steered, total


def main():
    # methods = ['pca', 'logistic', 'rfm']
    # methods = ['pca', 'rfm']
    # methods = ['rfm']
    # methods = ['pca']
    # methods = ['logistic']
    methods = ['mean_difference']

    # concepts = ['personalities', 'personas', 'moods', 'topophile']
    # concepts = ['fears', 'personalities', 'moods', 'places', 'personas']
    # concepts = ['fears']
    # concepts = ['personalities']
    # concepts = ['moods']
    # concepts = ['places']
    concepts = ['personas']

    MODEL_VERSION = '3.1'
    MODEL_SIZE = '70B'
    # VERSIONS = [4, 5]#[1, 2, 3]
    VERSIONS = [1, 2, 3, 4, 5]

    for VERSION in VERSIONS:

        VERSION_LABEL = ''
        if VERSION >= 2: 
            VERSION_LABEL = f'_v{VERSION}'
        results = {}
        for method in methods: 
            results[method] = []
            for concept in concepts: 
                fname = f'csvs/{method}_{concept}_gpt4o_outputs_500_concepts_llama_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.csv'
                # fname = f'csvs/{method}_{concept}_gpt4o_outputs_500_concepts_llama_3.3_english_only_v4.csv'
                # fname = f'csvs/{method}_{concept}_gpt4o_outputs_500_concepts_llama_3.3_english_only_v3.csv'
                # fname = f'csvs/{method}_{concept}_gpt4o_outputs_500_concepts_llama_3.3_english_only_v2.csv'
                # fname = f'csvs/{method}_{concept}_gpt4o_outputs_500_concepts_llama_3.3_english_only.csv'
                # fname = f'csvs/{method}_{concept}_gpt4o_outputs_500_concepts_llama_3.1_v2.csv'
                # fname = f'csvs/{method}_{concept}_gpt4o_outputs.csv'
                
                steered, total = readfile(fname)
                results[method].append((concept, steered, total))
        # print(results)
        for method in results: 
            # print(method, results[method])
            total = 0
            steered = 0
            for c, s, t in results[method]:
                total += t
                steered += s
            print(method, steered, total, steered/total)
        pass

if __name__ == "__main__":
    main()