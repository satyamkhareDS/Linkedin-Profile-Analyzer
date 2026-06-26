import streamlit as st
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(
    page_title="LinkedIn Profile Analyzer",
    page_icon="💼",
    layout="wide"
)

st.title("💼 LinkedIn Profile Analyzer & Career Predictor")
st.markdown("---")

# User Inputs
name = st.text_input("👤 Name")

position = st.text_input("💼 Current Position")

description = st.text_area("📝 Profile Summary")

experience = st.text_area("🏢 Experience")

skills = st.text_area(
    "🛠 Skills (comma separated)",
    placeholder="Python, Machine Learning, SQL"
)

if st.button("🚀 Analyze Profile"):

    profile_text = (
        str(description)
        + " "
        + str(experience)
        + " "
        + str(position)
        + " "
        + str(skills)
    )

    # Prediction
    vector = vectorizer.transform([profile_text])

    prediction = model.predict(vector)[0]

    st.success(f"🎯 Predicted Career Category: {prediction}")

    # Profile Score
    score = 0

    if len(description) > 100:
        score += 30

    if len(experience) > 100:
        score += 30

    skill_count = len([x for x in skills.split(",") if x.strip()])

    if skill_count >= 5:
        score += 40
    elif skill_count >= 3:
        score += 20

    score = min(score, 100)

    st.subheader("📊 Profile Strength Score")

    st.progress(score / 100)

    st.metric("Score", f"{score}/100")

    # Suggestions
    st.subheader("💡 Improvement Suggestions")

    if score < 70:

        st.warning("Your profile can be improved.")

        if len(description) < 100:
            st.write("✅ Add a detailed profile summary")

        if len(experience) < 100:
            st.write("✅ Add more experience details")

        if skill_count < 5:
            st.write("✅ Add more technical skills")

    else:
        st.success("Excellent LinkedIn Profile!")

    # Skill Recommendations
    st.subheader("🔥 Recommended Skills")

    recommendations = [
        "Python",
        "Machine Learning",
        "Data Analysis",
        "SQL",
        "Power BI",
        "Deep Learning",
        "Cloud Computing",
        "Git & GitHub"
    ]

    existing_skills = [
        s.strip().lower()
        for s in skills.split(",")
    ]

    for skill in recommendations:
        if skill.lower() not in existing_skills:
            st.write("✔", skill)

    # Headline Generator
    st.subheader("✨ Suggested LinkedIn Headlines")

    st.info(
        f"{position} | AI & Data Enthusiast | "
        f"Problem Solver | Lifelong Learner"
    )

    st.info(
        f"{position} | Machine Learning | "
        f"Python | SQL | Analytics"
    )

    # Project Suggestions
    st.subheader("📂 Recommended Projects")

    projects = [
        "LinkedIn Profile Analyzer",
        "Credit Scoring Model",
        "Smart Placement Predictor",
        "Fake News Detection",
        "Customer Churn Prediction",
        "Sales Forecast Dashboard",
        "Resume Screening System"
    ]

    for p in projects:
        st.write("📌", p)

    # Career Roadmap
    st.subheader("🛣 Career Roadmap")

    roadmap = [
        "Learn Advanced Python",
        "Master SQL",
        "Build ML Projects",
        "Learn Power BI",
        "Create GitHub Portfolio",
        "Apply for Internships",
        "Prepare for Interviews"
    ]

    for step in roadmap:
        st.write("➡", step)

    st.balloons()
