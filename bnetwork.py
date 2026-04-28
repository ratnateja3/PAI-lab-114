from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
data = pd.DataFrame({
    'Rain': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'TrafficJam': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
    'ArriveLate': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No']
})
data = data.astype('category')
print("Dataset:\n", data)
model = DiscreteBayesianNetwork([
    ('Rain', 'TrafficJam'),
    ('TrafficJam', 'ArriveLate')
])

model.fit(data, estimator=MaximumLikelihoodEstimator)

print("\nConditional Probability Distributions (CPDs):")
for cpd in model.get_cpds():
    print(cpd)

inference = VariableElimination(model)

# Query: P(ArriveLate | Rain = Yes)
result = inference.query(
    variables=['ArriveLate'],
    evidence={'Rain': 'Yes'}
)

print("\nQuery Result P(ArriveLate | Rain=Yes):")
print(result)