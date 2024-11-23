import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
data = pd.read_csv('/content/FOOTBALL DATASET.csv')
categorical_columns = ['Team', 'Player Position', 'Goal Hitted Position', 'Weather Conditions',
                       'Home/Away Indicator', 'Player Behaviour', 'Pitch Condition']
label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])
X = data.drop(['Substitute Player', 'Player Name'], axis=1)
y = data['Substitute Player']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
model = XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='mlogloss')
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Accuracy: {accuracy:.4f}")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Substituted', 'Substituted'], yticklabels=['Not Substituted', 'Substituted'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
top_players = data.sort_values(by='Score of Player Based on Field Conditions', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_players, x='Player Name', y='Score of Player Based on Field Conditions', palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Players Based on Field Conditions Score')
plt.ylabel('Score')
plt.xlabel('Player Name')
plt.show()
plt.figure(figsize=(8, 6))
sns.kdeplot(data=data, x=col, hue='Substitute Player', fill=True, alpha=0.6, palette='muted')
plt.title(f'Overlapping Distribution of {col} by Substitute Player')
plt.show()
substitute_indices = X_test.index[y_pred == 1]

print("\nPlayers predicted to be substituted:")
for idx in substitute_indices:
    print(data.loc[idx, 'Player Name'])
