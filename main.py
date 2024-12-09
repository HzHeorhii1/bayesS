import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from bayes.nbcNursery import NBCDiscreteForNursery

###############################
#        FOR NURSERY
###############################
attributes = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
decision_classes = ["not_recom", "recommend", "very_recom", "priority", "spec_prior"]
data = pd.read_csv('nursery.data', names=attributes)

# Кодування категоріальних змінних
encoders = {col: LabelEncoder() for col in data.columns}
for column in data.columns:
    data[column] = encoders[column].fit_transform(data[column])

# Розділення на ознаки та мітки
X = data.drop('class', axis=1).values
y = data['class'].values

X_train = X
y_train = y

# Навчання моделі
model = NBCDiscreteForNursery(alpha=1)
model.fit(X_train, y_train)

# Отримуємо закодовані значення для кожного класу
encoded_classes = {cls: encoders['class'].transform([cls])[0] for cls in decision_classes}

def calculate_conditional_probability_with_laplace(attribute, value, class_label):
    attribute_index = attributes.index(attribute)

    # Отримуємо кодування значення
    encoded_value = encoders[attribute].transform([value])[0]

    # Частота
    numerator = model.classFeatureCounts_[class_label][attribute_index].get(encoded_value, 0) + model.alpha_
    denominator = model.label_counts_[class_label] + len(model.featureCounts_[attribute_index]) * model.alpha_

    return numerator / denominator

def calculate_conditional_probability_without_laplace(attribute, value, class_label):
    attribute_index = attributes.index(attribute)

    # Отримуємо кодування значення
    encoded_value = encoders[attribute].transform([value])[0]

    # Частота
    numerator = model.classFeatureCounts_[class_label][attribute_index].get(encoded_value, 0)
    denominator = model.label_counts_[class_label]

    return numerator / denominator if denominator > 0 else 0

# Обчислення умовних ймовірностей для всіх класів
conditional_probabilities_with = {cls: {} for cls in decision_classes}
conditional_probabilities_without = {cls: {} for cls in decision_classes}

attributes_values = {
    'parents': 'usual',
    'has_nurs': 'less_proper',
    'form': 'incomplete',
    'children': '2',
    'housing': 'less_conv',
    'finance': 'convenient',
    'social': 'problematic',
    'health': 'recommended',
}

for cls in decision_classes:
    for attribute, value in attributes_values.items():
        conditional_probabilities_with[cls][attribute] = calculate_conditional_probability_with_laplace(attribute, value, encoded_classes[cls])
        conditional_probabilities_without[cls][attribute] = calculate_conditional_probability_without_laplace(attribute, value, encoded_classes[cls])

prior_probabilities = {cls: model.label_counts_[encoded_classes[cls]] / model.total_count_ for cls in decision_classes}

# Обчислення загальних ймовірностей для всіх класів
prob_without_laplace = {}
prob_with_laplace = {}
for cls in decision_classes:
    prob_w_o = prior_probabilities[cls]
    prob_w_l = prior_probabilities[cls]
    for attribute, value in attributes_values.items():
        prob_w_o *= conditional_probabilities_without[cls][attribute]
        prob_w_l *= conditional_probabilities_with[cls][attribute]
    prob_without_laplace[cls] = prob_w_o
    prob_with_laplace[cls] = prob_w_l

results_with = np.array([prob_with_laplace[cls] for cls in decision_classes])

normalized_with = results_with / results_with.sum()

print("\nwłączając poprawkę La- Place’a")
print(results_with)
print("Dzieląc każdy wynik przez sumę wszystkich uzyskamy prawdopodobieństwa")
print(normalized_with)


###############################
#        FOR WINE
###############################
