
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd

# 1. Load Data
# df = pd.read_csv('ratings.csv') # Must have userID, itemID, rating
# Load built-in Movielens-100k data (downloading sample)
data = Dataset.load_builtin('ml-100k')

# 2. Train SVD (Singular Value Decomposition)
trainset, testset = train_test_split(data, test_size=0.25)

algo = SVD()
algo.fit(trainset)

# 3. Predict
predictions = algo.test(testset)

# 4. Evaluate
from surprise import accuracy
accuracy.rmse(predictions)

# Example prediction
uid = str(196)  # raw user id (as in the ratings file)
iid = str(302)  # raw item id (as in the ratings file)
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
