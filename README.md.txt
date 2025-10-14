# Iris Flower Classification using Scikit-Learn

So, This project demonstrates how to build a **Machine Learning model using *Support Vector Machine (SVM)* and *K-Nearest Neighbors (KNN)* to classify the famous *Iris flower dataset* into three species — Setosa, Versicolor, and Virginica.

---

# Dataset
The *iris dataset* is a built-in dataset from `scikit-learn`.

Each sample contains:
-  Sepal Length  
-  Sepal Width  
-  Petal Length  
-  Petal Width  
-  Target class (Iris-setosa / Iris-versicolor / Iris-virginica)

---

# Model Workflow
1. Load dataset from `sklearn.datasets`
2. Split into *train* and *test* sets
3. Handle missing values using `SimpleImputer`
4. Train models:
   - **SVM (Support Vector Machine)** classifies data by finding an optimal line or hyperplane that maximizes the distance between each class in an N-dimensional space
   - **KNN (K-Nearest Neighbors)** makes a prediction based on the average of the values closest to the query point
5. Compare their accuracies #by using accuracy_score and confusion matrix we can predict the accuracy in numerical way.

---

# Results

| Model | Accuracy |
|--------|-----------|
| SVM | 0.97 |
| KNN | 0.96 |

*(Values will vary slightly on re-run)*

---

# Visualization

The plot below compares the accuracies of both models:

```python
plt.figure(figsize=(6,4))
sns.barplot(x=models, y=accuracies, hue=models, palette='Set2', legend=False)
plt.title("Model Accuracy Comparison")
plt.ylim(0.9, 1.0)

plt.ylim(0.9, 1.0)

# Save the figure as PNG
plt.savefig("model_accuracy.png", bbox_inches='tight')

plt.show()

