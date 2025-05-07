# EC_proj

Engenharia do Conhecimento 2023/2024

Objetives:

O1.

    For this objectve, it is necessary to know if a person has died considering the attributes in the data provided. Students can add new features whenever they consider it appropriate.
    Students should provide the best possible classification models using whatever method covered in class [Remember: everything else being similar, the simplest models should be preferred].
    Models should examine different hyperparameters and select the best one

O2.

    Can we confidently predict the AGE of the subject given the other attributes?

O3.

    Considering only the people who died, can we confidently predict the age of the people who did and didn't have Covid?

O4.

    What are the most significant features in the best models obtained above (in O1, O2 and O3)?

 

 

Processing notes

    Your notebook should be prepared to test the models with the independent test set.  Failling to do so, will significantly penalise the project.  Mock-ups of test data files are available for the effect, eg. proj-test-data.csv and project-test-class.csv.
    For more information about the data of each column see Section 2.

Evaluation

    The work will be evaluated taking into account:
        the description of the steps taken to achieve the objectives
        the implementation according to the mentioned description
        the evaluation of the best models performance obtained in an independent test set
        an individual quizz test on the developed project, hands on in the lab computer (in the last TP class)

Submission

Students should submit a ZIP file (no other compression algorithms allowed) with 2 documents

    PDF - Report with 6 pages MAXIMUM [aprox. 3000 words], containing
        Header:
            Group number
            Student names and ids
            how many hours each student contributed to the project
        Report text
            Introduction and Goals - brief description of the task, data and goals.
            Data processing - Briefly describe all data processing steps
            Variable selection - Describe the approach followed for variable selection
            Model results - should include a table with the statistics for the best models tested
            Hyperparameter tuning - briefly describe which parameters were tuned and how
            Discussion and Conclusions
        all sections can be supported by graphics
    Jupyter Notebook with the source code and attached results that led to the report. This Notebook with outputs should never be larger than 12 pages [penalties will be applied if size is exceeded]

Submission

    Submit the zip file on Moodle with one submission per group, up to May 26 at 12:00

 

Section 2:

 

        Attributes

    USMER: Indicates whether the patient was treated in medical units. 1 - yes, 2 - no.
    MEDICAL_UNIT: type of institution of the National Health System that provided the care. from 1 to 13.
    SEX: female or male: 1 - female, 2 - male.
    PATIENT_TYPE: hospitalized or not hospitalized. 1 - hospitalized, 2 - not hospitalized
    DATE_DIED: If the patient died indicate the date of death, and 9999-99-99 otherwise.
    INTUBED: whether the patient was connected to the ventilator. 1 - yes, 2 - no.
    PNEUMONIA: whether the patient already have air sacs inflammation or not. 1 - yes, 2 - no.
    AGE: of the patient.
    PREGNANT: whether the patient is pregnant or not. 1 - yes, 2 - no.
    DIABETES: whether the patient has diabetes or not. 1 - yes, 2 - no.
    COPD: Indicates whether the patient has Chronic obstructive pulmonary disease or not. 1 - yes, 2 - no.
    ASTHMA: whether the patient has asthma or not. 1 - yes, 2 - no.
    INMSUPR: whether the patient is immunosuppressed or not. 1 - yes, 2 - no.
    HIPERTENSION: whether the patient has hypertension or not. 1 - yes, 2 - no.
    OTHER_DISEASE: whether the patient has other disease or not. 1 - yes, 2 - no.
    CARDIOVASCULAR: whether the patient has heart or blood vessels related disease. 1 - yes, 2 - no.
    OBESITY: whether the patient is obese or not. 1 - yes, 2 - no.
    RENAL_CHRONIC: whether the patient has chronic renal disease or not. 1 - yes, 2 - no.
    TOBACCO: whether the patient is a tobacco user. 1 - yes, 2 - no.
    TEST_RESULT: covid test findings. Values 1-3 mean that the patient was diagnosed with covid in different degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.
    ICU: Indicates whether the patient had been admitted to an Intensive Care Unit. 1 - yes, 2 - no.

In the Boolean features, 1 means "yes" and 2 means "no". values as 97, 98 and 99 are missing data.
