import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------- 1. DATA GENERATION ----------
np.random.seed(42)
n = 200

data = pd.DataFrame({
    "study_hours": np.random.normal(6, 2, n),
    "attendance": np.random.randint(50, 100, n),
    "marks": np.random.normal(65, 10, n),
    "gender": np.random.choice(["Male", "Female"], n)
})

print("\n--- DATA SAMPLE ---")
print(data.head())


# ---------- 2. FREQUENCY DISTRIBUTION ----------
print("\n--- FREQUENCY DISTRIBUTION (MARKS) ---")
freq = pd.cut(data["marks"], bins=5).value_counts()
print(freq)


# ---------- 3. DATA VISUALIZATION ----------
plt.figure()
plt.hist(data["marks"], bins=10)
plt.title("Marks Histogram")
plt.show()

plt.figure()
plt.boxplot(data["marks"])
plt.title("Boxplot of Marks")
plt.show()

data["gender"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Gender Distribution")
plt.show()


# ---------- 4. CENTRAL TENDENCY ----------
mean = data["marks"].mean()
median = data["marks"].median()
mode = data["marks"].mode()[0]

print("\n--- CENTRAL TENDENCY ---")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)


# ---------- 5. DISPERSION ----------
variance = data["marks"].var()
std_dev = data["marks"].std()
iqr = stats.iqr(data["marks"])
cv = std_dev / mean

print("\n--- DISPERSION ---")
print("Variance:", variance)
print("Std Deviation:", std_dev)
print("IQR:", iqr)
print("Coefficient of Variation:", cv)


# ---------- 6. SKEWNESS & KURTOSIS ----------
skew = stats.skew(data["marks"])
kurt = stats.kurtosis(data["marks"])

print("\n--- SHAPE OF DISTRIBUTION ---")
print("Skewness:", skew)
print("Kurtosis:", kurt)


# ---------- 7. CORRELATION ----------
corr, _ = stats.pearsonr(data["study_hours"], data["marks"])
print("\n--- CORRELATION ---")
print("Correlation between Study Hours & Marks:", corr)

plt.scatter(data["study_hours"], data["marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Correlation Scatter Plot")
plt.show()


# ---------- 8. LINEAR REGRESSION ----------
slope, intercept, r, p, _ = stats.linregress(
    data["study_hours"], data["marks"]
)

print("\n--- LINEAR REGRESSION ---")
print("Regression Equation: y =", slope, "x +", intercept)
print("R value:", r)
print("p-value:", p)

plt.scatter(data["study_hours"], data["marks"])
plt.plot(data["study_hours"], slope*data["study_hours"] + intercept)
plt.title("Regression Line")
plt.show()


# ---------- 9. PROBABILITY ----------
prob_above_70 = len(data[data["marks"] > 70]) / n
print("\n--- PROBABILITY ---")
print("P(Marks > 70):", prob_above_70)


# ---------- 10. PROBABILITY DISTRIBUTIONS ----------
x = np.linspace(40, 90, 100)
y = stats.norm.pdf(x, mean, std_dev)

plt.plot(x, y)
plt.title("Normal Distribution Curve")
plt.show()

binom = stats.binom.pmf(5, 10, 0.5)
pois = stats.poisson.pmf(3, 2)

print("\n--- DISTRIBUTIONS ---")
print("Binomial P(X=5):", binom)
print("Poisson P(X=3):", pois)


# ---------- 11. SAMPLING & CLT ----------
sample_means = []
for i in range(1000):
    sample = data["marks"].sample(30)
    sample_means.append(sample.mean())

plt.hist(sample_means, bins=30)
plt.title("Central Limit Theorem")
plt.show()


# ---------- 12. HYPOTHESIS TESTING ----------
t_stat, p_val = stats.ttest_1samp(data["marks"], 65)

print("\n--- HYPOTHESIS TEST ---")
print("t-statistic:", t_stat)
print("p-value:", p_val)

if p_val < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")


# ---------- 13. MINI CASE STUDY ----------
print("\n--- MINI CASE STUDY ---")
print("Does studying more increase marks?")
print("Correlation:", corr)
print("Slope:", slope)
print("p-value:", p)

if slope > 0 and p < 0.05:
    print("Conclusion: YES, studying helps.")
else:
    print("Conclusion: No significant effect.")

print("\n=== ANALYSIS COMPLETE ===")
