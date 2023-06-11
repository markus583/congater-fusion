datasets = ['QQP', 'MNLI', ]  # Add more dataset names as needed
values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

def generate_value_combinations(dataset_names, values):
    if not dataset_names:
        return [{}]

    current_dataset = dataset_names[0]
    remaining_datasets = dataset_names[1:]
    combinations = []

    for value in values:
        remaining_combinations = generate_value_combinations(remaining_datasets, values)
        for combination in remaining_combinations:
            combination[current_dataset] = value
        combinations.extend(remaining_combinations)

    # filter out combinations where len(datasets)-1 values are 0
    # return_combinations = []
    # for combination in combinations:
    #     # exclude combinations where len(datasets)-1 values are 0
    #     if list(combination.values()).count(0) == len(datasets) - 1:
    #         continue
    #     return_combinations.append(combination)
    return combinations


if __name__ == '__main__':
    value_combinations = generate_value_combinations(datasets, values)

    # Print the value combinations
    for combination in value_combinations:
        print(combination)
    print((value_combinations))
    # # to json
    # import json
    # with open('value_combinations.json', 'w') as f:
    #     json.dump(value_combinations, f)
    print(len(value_combinations))
