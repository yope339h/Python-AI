import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("C:/Users/ahmed/Documents/health_lifestyle_classification.csv")

print(df.info())
print("Columns:", df.columns)
print(df.head(10))

X = df.drop(columns=['target'])
y = df['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

print("test set eval")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("RMSE:", mean_squared_error(y_test, y_pred_test)**0.5)
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))

print("\nvalidation set eval")
print("Accuracy:", accuracy_score(y_val, y_pred_val))
print("MAE:", mean_absolute_error(y_val, y_pred_val))
print("MSE:", mean_squared_error(y_val, y_pred_val))
print("RMSE:", mean_squared_error(y_val, y_pred_val)**0.5)
print("Precision:", precision_score(y_val, y_pred_val))
print("Recall:", recall_score(y_val, y_pred_val))
print("F1 Score:", f1_score(y_val, y_pred_val))
conf_mat = confusion_matrix(y_val,y_pred_val)
print("confusion matrix:",conf_mat)

