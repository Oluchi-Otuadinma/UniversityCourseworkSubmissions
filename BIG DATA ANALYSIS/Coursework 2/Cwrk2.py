#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pyspark
from pyspark.sql import SparkSession

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pyspark.ml.feature import StringIndexer, OneHotEncoder, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.stat import Summarizer
from pyspark.sql import functions as F

from pyspark.sql.functions import col
from pyspark.sql import Row
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import roc_curve, auc

from pyspark.ml import Pipeline


# In[36]:


sc = pyspark.SparkContext(appName="DiabetesPrediction")

spark = SparkSession(sc)


# In[37]:


# Load data
path = "hdfs://dsm-master:9000/user/ootua001/cwrk2/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
df = spark.read.csv(path, header=True, inferSchema=True)

#Inspect schema
df.printSchema()


# In[38]:


# Copied category descriptions from the UCI repository
metadata_rows = [
    Row(variable_name='ID', type='Integer', description='Patient ID'),
    Row(variable_name='Diabetes_binary', type='Binary', description='0 = no diabetes, 1 = prediabetes or diabetes'),
    Row(variable_name='HighBP', type='Binary', description='0 = no high BP, 1 = high BP'),
    Row(variable_name='HighChol', type='Binary', description='0 = no high cholesterol, 1 = high cholesterol'),
    Row(variable_name='CholCheck', type='Binary', description='0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years'),
    Row(variable_name='BMI', type='Integer', description='Body Mass Index'),
    Row(variable_name='Smoker', type='Binary', description='0 = no, 1 = yes, Have you smoked at least 100 cigarettes in your entire life'),
    Row(variable_name='Stroke', type='Binary', description='0 = no, 1 = yes, (Ever told) you had a stroke'),
    Row(variable_name='HeartDiseaseorAttack', type='Binary', description='0 = no, 1 = yes, coronary heart disease (CHD) or myocardial infarction (MI)'),
    Row(variable_name='PhysActivity', type='Binary', description='0 = no, 1 = yes, physical activity in past 30 days - not including job'),
    Row(variable_name='Fruits', type='Binary', description='0 = no, 1 = yes, Consume fruit 1 or more times per day'),
    Row(variable_name='Veggies', type='Binary', description='0 = no, 1 = yes, Consume vegetables 1 or more times per day'),
    Row(variable_name='HvyAlcoholConsump', type='Binary', description='0 = no, 1 = yes, Heavy drinkers (adult men having more than 14 drinks per week)'),
    Row(variable_name='AnyHealthcare', type='Binary', description='0 = no, 1 = yes, Have any kind of health care coverage'),
    Row(variable_name='NoDocbcCost', type='Binary', description='0 = no, 1 = yes, Was there a time in the past 12 months when you needed to see a doctor but could not because of cost'),
    Row(variable_name='GenHlth', type='Integer', description='Health status scale 1 = excellent, 5 = poor'),
    Row(variable_name='MentHlth', type='Integer', description='Mental health (stress, depression) for how many days during the past 30 days was your mental health not good?'),
    Row(variable_name='PhysHlth', type='Integer', description='Physical health (illness, injury) for how many days during the past 30 days was your physical health not good?'),
    Row(variable_name='DiffWalk', type='Binary', description='0 = no, 1 = yes, Do you have serious difficulty walking or climbing stairs?'),
    Row(variable_name='Sex', type='Binary', description='0 = female, 1 = male'),
    Row(variable_name='Age', type='Integer', description='Age categories 1 = 18-24, 9 = 60-64, 13 = 80 or older'),
    Row(variable_name='Education', type='Integer', description='Education level 1 = Never attended school, 6 = College graduate'),
    Row(variable_name='Income', type='Integer', description='Income scale 1-8 1 = less than $10,000, 8 = $75,000 or more')
]

# Create the metadata DataFrame
metadata_df = spark.createDataFrame(metadata_rows)

# Show the metadata DataFrame
metadata_df.show(truncate=False)


# In[42]:


df.head()


# In[47]:


# Calculate the mean, standard deviation, and min/max values for the dataframe
mean_values = df.select([F.mean(col).alias(col) for col in df.columns])
std_values = df.select([F.stddev(col).alias(col) for col in df.columns])
min_values = df.select([F.min(col).alias(col) for col in df.columns])
max_values = df.select([F.max(col).alias(col) for col in df.columns])

# Show the results
mean_values.show()


# In[48]:


std_values.show()


# In[49]:


min_values.show()


# In[50]:


max_values.show()


# In[51]:


df.count()


# In[52]:


# 4. Drop rows with missing values
df = df.dropna()


# In[53]:


df.count()


# In[54]:


# List all feature columns (excluding the target 'Diabetes_binary')
feature_cols = [col for col in df.columns if col != 'Diabetes_binary']

# Assemble features into a single vector column
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
feature_names = assembler.getInputCols()
df_final = assembler.transform(df)

# Select features and label
df_final = df_final.select("features", df_final["Diabetes_binary"].alias("label"))


# In[55]:



# Train-test split
train, test = df_final.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LogisticRegression()
lr_model = lr.fit(train)


# In[56]:


def evaluate_model_with_plots(model, test_data, save_prefix="model_eval", feature_names=None):
    
    # Make predictions on the test data
    predictions = model.transform(test_data).select("prediction", "label", "probability")
    
    # Create confusion matrix
    confusion_matrix = predictions.groupBy("label", "prediction").count().collect()
    
    # Convert the confusion matrix into a dictionary for easier reading
    cm_dict = {}
    for row in confusion_matrix:
        label = row["label"]
        pred = row["prediction"]
        count = row["count"]
        cm_dict[(label, pred)] = count
    
    # Extract confusion matrix values
    tp = cm_dict.get((1.0, 1.0), 0)
    fp = cm_dict.get((0.0, 1.0), 0)
    tn = cm_dict.get((0.0, 0.0), 0)
    fn = cm_dict.get((1.0, 0.0), 0)
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print(f"True Positive (TP): {tp}")
    print(f"False Positive (FP): {fp}")
    print(f"True Negative (TN): {tn}")
    print(f"False Negative (FN): {fn}")
    
    # Compute additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Print additional metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    
    # Plot and save Confusion Matrix
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred: 0", "Pred: 1"], yticklabels=["True: 0", "True: 1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png")
    plt.show()
    
    # Evaluate model with AUC using the "probability" column
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability")
    auc_value = evaluator.evaluate(predictions)
    print(f"AUC: {auc_value}")
    
    # Plot ROC Curve
    # Extract true labels and probabilities for ROC curve
    true_labels = predictions.select("label").rdd.map(lambda row: row[0]).collect()
    probabilities = predictions.select("probability").rdd.map(lambda row: row[0][1]).collect()  # probability of class 1
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Plot and save ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_roc_curve.png")
    plt.show()
    
    # Feature Importance Plotting
    if hasattr(model, "featureImportances") and feature_names is not None:
        # Extract feature importances from the model
        importances = model.featureImportances.toArray()  # Convert to a numpy array
        
        #I had tried to one-hot encode prior but the category size kept exploding so it was removed
        print(f"DEBUG: Length of feature_names: {len(feature_names)}")
        print(f"DEBUG: Length of importances: {len(importances)}")

        if len(feature_names) != len(importances):
            raise ValueError(
                f"Mismatch in feature names and importance array lengths: "
                f"{len(feature_names)} names vs {len(importances)} importances"
            )

        # Create a DataFrame with feature names and their corresponding importances
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        # Plot and save the feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df)
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_feature_importance.png")
        plt.show()

        print(f"Feature importances saved as {save_prefix}_feature_importance.png")

        
    # Save evaluation summary to text file
    with open(f"{save_prefix}_summary.txt", "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(f"True Positive (TP): {tp}\n")
        f.write(f"False Positive (FP): {fp}\n")
        f.write(f"True Negative (TN): {tn}\n")
        f.write(f"False Negative (FN): {fn}\n\n")
        f.write(f"AUC: {auc_value:.4f}\n")
        f.write(f"ROC Curve AUC: {roc_auc:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1_score:.4f}\n")
        
    print(f" Plots saved as {save_prefix}_confusion_matrix.png and {save_prefix}_roc_curve.png")
    print(f" Summary saved as {save_prefix}_summary.txt")    
    
        # Final return dictionary
    result = {
        "Confusion Matrix": cm_dict,
        "AUC": auc_value,
        "ROC AUC": roc_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
    }
    
    
    if hasattr(model, "featureImportances") and feature_names is not None:
        result["Feature Importances"] = dict(zip(feature_names, importances.tolist()))
        
    # Return confusion matrix and AUC for further use
    return result


# In[57]:


# Logistic Regression
results_lr = evaluate_model_with_plots(lr_model, test, save_prefix="diabetes_logreg_eval", feature_names=feature_names)


# In[58]:


from pyspark.ml.classification import RandomForestClassifier

# Train model
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
rf_model = rf.fit(train)

# Evaluate Random Forest
results_rf = evaluate_model_with_plots(rf_model, test, save_prefix="diabetes_rf_eval", feature_names=feature_names)


# In[59]:


from pyspark.ml.classification import GBTClassifier

# Train model
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=50)
gbt_model = gbt.fit(train)

# Evaluate
results_gbt = evaluate_model_with_plots(gbt_model, test, save_prefix="diabetes_gbt_eval", feature_names=feature_names)


# In[60]:


# Extract the metrics
# Define models and metrics
model_names = ["Logistic Regression", "Random Forest", "GBT"]
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Gather results for each model
all_results = [results_lr, results_rf, results_gbt]
metric_values = [[res[metric] for metric in metrics] for res in all_results]

# --- Plot Comparison of All Models ---
x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(10, 6))
for i, (model, values) in enumerate(zip(model_names, metric_values)):
    plt.bar(x + i * width, values, width=width, label=model)

plt.xticks(x + width, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Evaluation Metrics Comparison")
plt.legend()
plt.tight_layout()
plt.show()

