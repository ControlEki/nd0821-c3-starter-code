# Model Card


## Model Details

This model is designed to classify individuals into salary categories, such as earning above or below a specified threshold (for example, $50K), using demographic and employment-related attributes from U.S. Census data.

- Model: RandomForestClassifier
- Training Framework: scikit-learn
- Random State: 42

## Training Data

The model is trained using the census.csv dataset, which contains demographic and employment information. Relevant features include workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Evaluation Data

The same Census dataset is divided into training and testing subsets using an 80/20 split. The 20% test portion is reserved for evaluating how well the model performs on previously unseen observation.

## Metrics

Model performance is assessed using weighted precision, recall, and F1 score. The results obtained during evaluation are:

- Precision: 0.74
- Recall: 0.64
- F1 Score: 0.69

## Ethical Considerations

The dataset contains sensitive demographic attributes, including race, sex, and native country. Using these characteristics for salary prediction may introduce or reinforce existing biases related to income and employment opportunities. As a result, model predictions may reflect inequalities present in the underlying data.

## Caveats and Recommendations

The dataset may contain an imbalance between the salary classes, which can affect the model's ability to accurately predict the less-represented class. Techniques such as oversampling or undersampling could be considered to reduce the impact of class imbalance.

Because the model is trained on U.S. Census data, its results may not generalize well to populations in other countries or regions. Additional training data that better represents the intended population should be considered when applying the model in different contexts.

The model should also be retrained periodically using more recent data to account for changes in economic conditions, employment patterns, and broader socio-economic trends.

Finally, predictions should be treated as one source of information rather than as the sole basis for significant decisions. Model outputs should be evaluated alongside other relevant evidence and, where appropriate, expert judgment.
