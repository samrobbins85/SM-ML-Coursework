# Usage: This program will download the data for you, there is no need to download it yourself
# Should you not be connected to the internet or the download is not working, the csv files should be stored
# in a directory named "anonymisedData"
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import urllib.request
from zipfile import ZipFile

# Download the files
print("Downloading the files")
try:
    url = "https://analyse.kmi.open.ac.uk/open_dataset/download"
    urllib.request.urlretrieve(url, "data.zip")
except:
    print(
        "Can't download files, ensure they are correctly placed as documented in the top of the python file"
    )
else:
    print("Finished Downloading")
    file_name = "data.zip"

    # opening the zip file in READ mode
    with ZipFile(file_name, "r") as zip:
        # extracting all the files
        print("Extracting all the files now...")
        zip.extractall(path="anonymisedData")
        print("Done!")


# Read in all the CSV files required
studentInfo = pd.read_csv("anonymisedData/studentInfo.csv")
studentVle = pd.read_csv("anonymisedData/studentVle.csv")
studentAssessment = pd.read_csv("anonymisedData/studentAssessment.csv")
assessments = pd.read_csv("anonymisedData/assessments.csv")

# Remove all the unneeded columns
assessments = assessments.drop(columns=["assessment_type", "date", "weight"])
studentAssessment = studentAssessment.drop(columns=["date_submitted", "is_banked"])
studentVle = studentVle.drop(columns=["id_site", "date"])
studentInfo = studentInfo.drop(columns=["region", "age_band"])

# Group the clicks by a student for a specific module and presentation over multiple dates
newVle = (
    studentVle.groupby(["code_module", "code_presentation", "id_student"])
    .agg(sum_clicks=("sum_click", "sum"))
    .reset_index()
)

# Merge the VLE results with the Student Info results
merged = newVle.merge(
    studentInfo,
    left_on=["code_module", "code_presentation", "id_student"],
    right_on=["code_module", "code_presentation", "id_student"],
)

# Remove the columns we don't need
merged = merged.drop(columns=["code_module", "code_presentation", "id_student"])
merged = merged.dropna()


# Replace the labels with numbers
to_replace = {"Withdrawn": 0, "Fail": 0, "Pass": 1, "Distinction": 1}
merged["final_result"] = merged["final_result"].map(to_replace)


# print(merged)
# sns.regplot(
#     x="sum_clicks", y="final_result", data=merged, marker="+", line_kws={"color": "red"}
# )
# plt.show()


studentInfo = merged

# Replace each of the imd band ranges with the average value
studentInfo = studentInfo.replace(to_replace=["90-100%", "90-100"], value=95)
studentInfo = studentInfo.replace(to_replace=["80-90%", "80-90"], value=85)
studentInfo = studentInfo.replace(to_replace=["70-80%", "70-80"], value=75)
studentInfo = studentInfo.replace(to_replace=["60-70%", "60-70"], value=65)
studentInfo = studentInfo.replace(to_replace=["50-60%", "50-60"], value=55)
studentInfo = studentInfo.replace(to_replace=["40-50%", "40-50"], value=45)
studentInfo = studentInfo.replace(to_replace=["30-40%", "30-40"], value=35)
studentInfo = studentInfo.replace(to_replace=["20-30%", "20-30"], value=25)
studentInfo = studentInfo.replace(to_replace=["10-20%", "10-20"], value=15)
studentInfo = studentInfo.replace(to_replace=["0-10%", "0-10"], value=5)

# Map the values given for gender to numbers
studentInfo = studentInfo.replace(to_replace=["F"], value=0)
studentInfo = studentInfo.replace(to_replace=["M"], value=1)

# Map the highest education levels to numbers
to_replace = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
studentInfo["highest_education"] = studentInfo["highest_education"].map(to_replace)


studentInfo = studentInfo.dropna()

# Select the features and labels
X = studentInfo[
    [
        "imd_band",
        "sum_clicks",
        "gender",
        "studied_credits",
        "num_of_prev_attempts",
        "highest_education",
    ]
]
y = studentInfo.final_result
# Split the testing and training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=0
)


# Logistic regression
print("Performing logistic regression")
logreg = LogisticRegression(solver="liblinear")
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Logistic regression generated the following confusion matrix")
print(cnf_matrix)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("")
# Decision Tree
print("Performing decision tree classification")
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Decision tree classification generated the following confusion matrix")
print(cnf_matrix)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
