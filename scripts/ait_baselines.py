"""TF-IDF on AIT-ADS windows. Train on 6 attack scenarios, test on 2 it has never seen."""
import json, random, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
rows = json.load(open("data/ait/windows.json"))
TEST = {"wilson", "russellmitchell"}          # unseen scenarios
train = [r for r in rows if r["scenario"] not in TEST]
test = [r for r in rows if r["scenario"] in TEST]
 
def balance(rs):                               # 97% of windows are "low", so keep it fair
    low = [r for r in rs if r["severity"] == "low"]
    rest = [r for r in rs if r["severity"] != "low"]
    return rest + random.sample(low, min(len(low), len(rest)))
 
random.seed(0)
train, test = balance(train), balance(test)
json.dump(test, open("data/ait/test_windows.json", "w"))   # same test set for every model
print(len(train), "train /", len(test), "test", collections.Counter(r["severity"] for r in test))
 
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"[A-Za-z_]+")
clf = LogisticRegression(max_iter=3000, class_weight="balanced")
clf.fit(vec.fit_transform([r["text"] for r in train]), [r["severity"] for r in train])
pred = clf.predict(vec.transform([r["text"] for r in test]))
print(classification_report([r["severity"] for r in test], pred, digits=3, zero_division=0))
 