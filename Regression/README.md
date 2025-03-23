# The MLVault 
---

## Should one use `numpy` arrays or `df` in `sklearn` ?
### ✅ When It's Okay to Use DataFrames Directly
- **Input features (`X`)**: You can pass a DataFrame.
- **Target variable (`y`)**: You can use a Series (or a single-column DataFrame).
- Scikit-learn will internally convert them to NumPy arrays anyway.
- Feature names from the DataFrame will be used in some places (like in `sklearn` >= 1.0, with `get_feature_names_out`).

**Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'age': [25, 32, 47],
    'income': [50000, 60000, 80000],
    'purchased': [0, 1, 1]
})

X = df[['age', 'income']]
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

### ⚠️ When You Might Want to Convert to NumPy
1. **Speed**: If you're doing heavy numerical computation, NumPy arrays are a bit faster.
2. **Custom Models or Algorithms**: Some lower-level or experimental tools might expect NumPy.
3. **Incompatibility with Some Pipelines or Libraries**: Rare, but sometimes downstream tools (e.g. older libraries or tools like ONNX) prefer NumPy arrays.
4. **No Column Names Required**: If you don't need column names (like when using PCA, clustering, etc.), NumPy arrays are fine.

**To convert:**
```python
X.values  # Or X.to_numpy()
```

---

### ✅ TL;DR
> You can safely use **pandas DataFrames** with most scikit-learn models. Conversion to NumPy is not required, but you can do it if you want slightly more speed or are using lower-level tools.

---
# Regression 
---

### **1. Linear Relationships**
Use these models when the relationship between features and the target variable is **linear** or approximately linear.

- **`LinearRegression`**: Standard linear regression with no regularization.
- **`Ridge`**: Linear regression with **L2 regularization** to handle multicollinearity.
- **`Lasso`**: Linear regression with **L1 regularization** for feature selection.
- **`ElasticNet`**: Combines **L1 and L2 regularization** for a balance between Ridge and Lasso.
- **`SGDRegressor`**: Linear regression optimized using **Stochastic Gradient Descent** (supports L1, L2, and ElasticNet penalties).
- **`BayesianRidge`**: Bayesian linear regression with **L2 regularization** and probabilistic predictions.
- **`ARDRegression`**: Bayesian linear regression with **Automatic Relevance Determination** for feature selection.

---

### **2. Non-Linear Relationships**
Use these models when the relationship between features and the target variable is **non-linear**.

- **`DecisionTreeRegressor`**: A single decision tree for non-linear regression.
- **`RandomForestRegressor`**: Ensemble of decision trees for robust non-linear regression.
- **`GradientBoostingRegressor`**: Sequentially builds trees to correct errors, suitable for complex non-linear relationships.
- **`SVR` (Support Vector Regression)**: Uses kernel functions (e.g., RBF, polynomial) to model non-linear relationships.
- **`KernelRidge`**: Kernelized version of Ridge Regression for non-linear data.
- **`GaussianProcessRegressor`**: Non-parametric, kernel-based regression with probabilistic predictions.
- **`MLPRegressor`**: Neural network-based regressor for highly non-linear relationships.

---

### **3. Large-Scale or Streaming Data**
Use these models when working with **large datasets** or **streaming data** where memory efficiency and incremental learning are important.

- **`SGDRegressor`**: Optimized for large datasets using Stochastic Gradient Descent.
- **`PassiveAggressiveRegressor`**: Online learning algorithm for large-scale or streaming data.
- **`MiniBatchKMeans`**: Not a regressor, but useful for clustering large datasets.

---

### **4. Robust Regression (Handling Outliers)**
Use these models when your data contains **outliers** or is noisy.

- **`HuberRegressor`**: Uses **Huber loss** to reduce the influence of outliers.
- **`TheilSenRegressor`**: Uses the **Theil-Sen estimator**, which is highly robust to outliers.
- **`RANSACRegressor`**: Fits a model to inliers while ignoring outliers.

---

### **5. Feature Selection**
Use these models when you want to **select important features** or reduce dimensionality.

- **`Lasso`**: Performs **L1 regularization** to shrink some coefficients to zero.
- **`ElasticNet`**: Combines **L1 and L2 regularization** for feature selection and handling multicollinearity.
- **`ARDRegression`**: Automatically prunes irrelevant features using **Automatic Relevance Determination**.

---

### **6. Probabilistic Predictions**
Use these models when you need **uncertainty estimates** or probabilistic predictions.

- **`BayesianRidge`**: Provides uncertainty estimates for predictions.
- **`ARDRegression`**: Similar to `BayesianRidge` but with feature selection.
- **`GaussianProcessRegressor`**: Provides full probabilistic predictions with confidence intervals.

---

### **7. Ensemble Methods**
Use these models when you want to combine multiple models for **improved accuracy** and robustness.

- **`RandomForestRegressor`**: Ensemble of decision trees for robust predictions.
- **`GradientBoostingRegressor`**: Sequentially builds trees to minimize errors.
- **`AdaBoostRegressor`**: Boosts the performance of weak regressors (e.g., decision trees).
- **`XGBoost`**, **`LightGBM`**, **`CatBoost`**: Advanced gradient boosting frameworks for high performance.

---

### **8. Custom Kernel-Based Regression**
Use these models when you need **kernel-based regression** for non-linear data.

- **`SVR` (Support Vector Regression)**: Uses kernel functions (e.g., RBF, polynomial) for non-linear regression.
- **`KernelRidge`**: Kernelized Ridge Regression for non-linear relationships.
- **`GaussianProcessRegressor`**: Non-parametric, kernel-based regression with probabilistic outputs.

---

### **9. Online Learning**
Use these models for **streaming data** or when you need to update the model incrementally.

- **`SGDRegressor`**: Supports incremental learning with partial_fit.
- **`PassiveAggressiveRegressor`**: Designed for online learning.
- **`MLPRegressor`**: Can be updated incrementally with partial_fit.

---

### **Summary Table**

| **Situation**                     | **Recommended Models**                                                                 |
|-----------------------------------|---------------------------------------------------------------------------------------|
| **Linear Relationships**          | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDRegressor`, `BayesianRidge`   |
| **Non-Linear Relationships**      | `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `SVR`, `KernelRidge`, `GaussianProcessRegressor`, `MLPRegressor` |
| **Large-Scale or Streaming Data** | `SGDRegressor`, `PassiveAggressiveRegressor`                                          |
| **Robust Regression (Outliers)**  | `HuberRegressor`, `TheilSenRegressor`, `RANSACRegressor`                              |
| **Feature Selection**             | `Lasso`, `ElasticNet`, `ARDRegression`                                                |
| **Probabilistic Predictions**     | `BayesianRidge`, `ARDRegression`, `GaussianProcessRegressor`                          |
| **Ensemble Methods**              | `RandomForestRegressor`, `GradientBoostingRegressor`, `AdaBoostRegressor`, `XGBoost`, `LightGBM`, `CatBoost` |
| **Kernel-Based Regression**       | `SVR`, `KernelRidge`, `GaussianProcessRegressor`                                      |
| **Online Learning**               | `SGDRegressor`, `PassiveAggressiveRegressor`, `MLPRegressor`                          |

---

### **Key Takeaways**
- Choose **linear models** for simple, linear relationships.
- Use **non-linear models** (e.g., tree-based, kernel-based) for complex relationships.
- For **large datasets**, prefer iterative models like `SGDRegressor`.
- Use **robust models** when your data has outliers.
- For **feature selection**, consider `Lasso` or `ElasticNet`.
- If you need **uncertainty estimates**, go for Bayesian or Gaussian Process models.
- For **high accuracy**, use ensemble methods like `RandomForestRegressor` or `GradientBoostingRegressor`.

Always validate your choice using cross-validation and appropriate evaluation metrics!