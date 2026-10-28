EduMentor AI - An Intelligent Personalised Learning and Early Academic Risk Prediction System
Artificial Intelligence Project - Vellore Institute of Technology
Team: Shivam Prakash (24BDS0340), Devarsh Singh (24BAI0252)

------------------------------------------------------------
1. WHAT THIS FOLDER CONTAINS
------------------------------------------------------------
generate_dataset.py   Creates the synthetic student dataset (data/students.csv, 1,500 students).
train_model.py        Trains and compares 4 classifiers (Logistic Regression, Decision Tree, Random Forest,
                      Gradient Boosting), runs the feature ablation, saves figures.
experiments2.py       Hyper-parameter tuning, class-imbalance strategies (class weights, oversampling, SMOTE),
                      probability calibration (Platt), threshold/risk-band selection, exact Shapley-value
                      comparison, fairness-audit demonstration. Writes the final model outputs/web_model.json.
make_diagram.py       Draws the system architecture figure.
build_web.py          Builds web/index.html (student view + faculty dashboard) from web/template.html.
app.py                Flask + SQLite backend: sign-up/login (hashed passwords), saved snapshots, risk history,
                      faculty cohort from the database.
test_app.py           16 automated tests for the backend.
validate_uci.py       Validation on the real UCI Student Performance dataset (needs the dataset CSV).
user_study/           Pilot user-study protocol, SUS scoring script, response template.
run_all.py            Runs every step above in the correct order with one command.
data/, outputs/, figures/, web/   Generated data, result tables (CSV/JSON), figures and the web page.
                      They are already included, so results can be viewed without re-running anything.

------------------------------------------------------------
2. REQUIREMENTS
------------------------------------------------------------
Python 3.10 or newer (developed and tested with Python 3.12).
Install the libraries once:
    pip install -r requirements.txt
(numpy, pandas, scikit-learn, matplotlib, scipy, flask. "ucimlrepo" is needed only for validate_uci.py
without a local CSV file.)

------------------------------------------------------------
3. HOW TO RUN (all commands are typed inside this folder)
------------------------------------------------------------
A) Run everything with one command (about 1-2 minutes):
       python run_all.py
   This regenerates the dataset, trains and evaluates all models, runs the experiments, rebuilds the
   web page and runs the backend tests. It ends with "ALL TESTS PASSED".

B) Or run the steps one by one, in this order:
       python generate_dataset.py      -> data/students.csv
       python train_model.py           -> outputs/model_comparison.csv, ablation.csv, figures/
       python experiments2.py          -> outputs/tuning.csv, imbalance.csv, calibration.csv,
                                          bands_test.csv, fairness_demo.csv, exp2_results.json, web_model.json
       python build_web.py             -> web/index.html
       python test_app.py              -> backend tests
   (experiments2.py must run after train_model.py because it overwrites outputs/web_model.json with the
   calibrated model.)

C) View the application
   Option 1 - standalone page (no server): double-click web/index.html. Student view and faculty dashboard
              work immediately; the account/history panel stays hidden.
   Option 2 - with accounts, saved history and faculty cohort from the database:
              Windows (PowerShell):   $env:FACULTY_CODE="choose-a-code"; python app.py
              Linux / macOS:          FACULTY_CODE=choose-a-code python app.py
              then open  http://127.0.0.1:5000
              Register a student account and save snapshots; register a faculty account using the faculty
              code you chose (default if none is set: faculty123).

D) Validation on real data (optional, requires the dataset)
       python validate_uci.py path/to/student-mat.csv
   Download "Student Performance" from the UCI Machine Learning Repository (student.zip, file student-mat.csv).
   Results are written to outputs/uci_results.csv.

E) Pilot user study (optional)
   Follow user_study/PROTOCOL.md, save answers in a CSV like user_study/responses_template.csv, then run
       python user_study/score_sus.py responses.csv

------------------------------------------------------------
4. EXPECTED RESULTS (fixed random seed, so they are reproducible)
------------------------------------------------------------
Final model: calibrated logistic regression, test ROC-AUC about 0.972.
Risk bands: Low below 0.14, Medium 0.14 to 0.43, High 0.43 and above.
Test set (300 students): Low band holds 1 of 72 at-risk students; Medium+High recall 0.986.
Backend tests: "ALL TESTS PASSED" (16 checks).

------------------------------------------------------------
5. IMPORTANT NOTES
------------------------------------------------------------
* The main dataset is SYNTHETIC (simulated students). Results show that the pipeline works; they are not
  measurements on real students. Real-data validation is provided through validate_uci.py.
* The fairness audit uses a simulated group attribute to demonstrate the audit tooling.
* The backend is a demonstration. For real student data it would need HTTPS, a managed secret key and an
  institutional privacy review. The database file edumentor.db is created on first run of app.py.
* AI assistance: the code and report draft were produced with AI assistance (Claude, Anthropic). The full
  disclosure is in Section 9 (Contribution Statement) of the project report.
