import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load(path):
    texts, sev = [], []
    for line in open(path):
        msgs = json.loads(line)["messages"]
        texts.append(msgs[0]["content"].split("\n\n", 1)[1])   # the alert text
        sev.append(json.loads(msgs[1]["content"])["severity"])  # the label
    return texts, sev

train_x, train_y = load("data/mlx/train.jsonl")
test_x, test_y = load("data/mlx/test.jsonl")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(vec.fit_transform(train_x), train_y)
pred = clf.predict(vec.transform(test_x))

print(f"TF-IDF severity accuracy: {accuracy_score(test_y, pred):.1%}")
print(classification_report(test_y, pred, digits=3))