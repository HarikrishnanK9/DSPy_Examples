
import dspy
import math

# Step 1: Create a dataset of Q&A pairs

qa_pairs = [
    ("What is the capital of Japan?", "Tokyo"),
    ("Which element has the atomic number 1?", "Hydrogen"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the fastest land animal?", "Cheetah"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the square root of 64?", "8"),
    ("In which year did the Titanic sink?", "1912"),
    ("What is the hardest natural substance?", "Diamond"),
    ("Who developed the theory of relativity?", "Albert Einstein"),
    ("What is the largest ocean on Earth?", "Pacific Ocean")
]

# Step 2: Convert into DSPy Examples

dataset = [
    dspy.Example(question=q, answer=a).with_inputs("question")
    for q, a in qa_pairs
]

# Step 3: Train/validation split (20% / 80%)

split_index = math.ceil(len(dataset) * 0.2)  # 20% train
trainset = dataset[:split_index]
valset = dataset[split_index:]

print(f"Total examples: {len(dataset)}")
print(f"Training set size: {len(trainset)}")
print(f"Validation set size: {len(valset)}")

# Step 4: Define a Signature

class QA(dspy.Signature):
    """Answer the given question accurately."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="A short, factual answer.")

# Step 5: Configure a Language Model

lm = dspy.LM('ollama_chat/phi3', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

# Step 6: Create a predictor module

class QAPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(QA)

    def forward(self, question):
        return self.predict(question=question)

# Step 7: Define the evaluation metric

def exact_match_metric(gold, pred, trace=None):
    """
    Returns True if the predicted answer matches the gold answer (case-insensitive).
    """
    return pred.answer.strip().lower() == gold.answer.strip().lower()

# Step 8: Optimize with BootstrapFewShot

pipeline = QAPredictor()
optimizer = dspy.teleprompt.BootstrapFewShot(
    metric=exact_match_metric,
    max_bootstrapped_demos=3
)

optimized_pipeline = optimizer.compile(pipeline, trainset=trainset)

# Step 9: Test on validation set

print("\nValidation results:")
for example in valset:
    prediction = optimized_pipeline(question=example.question)
    print(f"Q: {example.question}")
    print(f"Predicted: {prediction.answer} | Gold: {example.answer}")
    print("-" * 40)
